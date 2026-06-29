# Description: Object-store data source: reads Parquet/CSV via DuckDB.
# Description: One reader for local files, S3/MinIO, GCS, and Azure Data Lake.

"""DuckDB-backed data source over open formats and object storage.

Reads Parquet or CSV from a single URI whose scheme selects the backend:
``file://`` (or a bare path), ``s3://`` (also MinIO/Ceph), ``gs://`` / ``gcs://``,
and ``az://`` (Azure Data Lake Gen2). The local path is the default and needs no
credentials. Cloud schemes pull DuckDB's httpfs/azure extension at runtime and
read credentials from the standard environment variables for that cloud.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from scry.data.sources.base import DataSource, normalize_record


def _scheme_of(uri: str) -> str:
    return uri.split("://", 1)[0].lower() if "://" in uri else "file"


def _format_of(uri: str) -> str:
    return "csv" if uri.lower().rstrip("/").endswith(".csv") else "parquet"


def _q(value: str) -> str:
    """Quote a value as a DuckDB string literal (SET statements take no params)."""
    return "'" + str(value).replace("'", "''") + "'"


def _features_config() -> dict[str, Any]:
    """Parse ``config/features.yaml`` and return its contents.

    Checks ``SCRY_FEATURES_PATH``, then searches upward from this file and the
    working directory for ``config/features.yaml``. Returns an empty dict when no
    config is found. This is the source's own profile resolution; it intentionally
    uses ``SCRY_FEATURES_PATH`` so coverage and the ``fetch_metrics`` profile
    filter agree on a profile's feature set.
    """
    import yaml

    candidates: list[Path] = []
    env_path = os.environ.get("SCRY_FEATURES_PATH")
    if env_path:
        candidates.append(Path(env_path))
    here = Path(__file__).resolve()
    candidates.extend(parent / "config" / "features.yaml" for parent in here.parents)
    candidates.append(Path.cwd() / "config" / "features.yaml")

    for path in candidates:
        if path.is_file():
            with open(path) as f:
                return yaml.safe_load(f) or {}
    return {}


def _load_profile_metrics(profile: str) -> list[str]:
    """Load the combined numerical+categorical metric names for one profile.

    Returns an empty list (no filter) when the config or the profile is absent.
    """
    p = _features_config().get("profiles", {}).get(profile)
    if not p:
        return []
    return list(p.get("numerical_features", [])) + list(p.get("categorical_features", []))


def _load_all_profiles() -> dict[str, dict[str, Any]]:
    """Map every profile name to its description and combined feature-name list."""
    out: dict[str, dict[str, Any]] = {}
    for name, p in _features_config().get("profiles", {}).items():
        out[name] = {
            "description": p.get("description", ""),
            "features": list(p.get("numerical_features", []))
            + list(p.get("categorical_features", [])),
        }
    return out


# A series is "live" when its last sample is within this many inferred intervals
# of the reference time.
_FRESH_K = 2


class ObjectStoreSource(DataSource):
    """Reads metric Parquet/CSV from a URI via DuckDB.

    Args:
        uri: Glob or path to the data, for example
            ``"data/metrics/**/*.parquet"``, ``"s3://bucket/metrics/**/*.parquet"``,
            or ``"az://container/metrics/**/*.parquet"``.
        hive_partitioning: Treat ``key=value`` path segments as columns (Parquet only).
        data_format: ``"parquet"`` or ``"csv"``; inferred from the URI when omitted.
        connection_string: Optional Azure storage connection string. Cloud
            credentials otherwise come from the environment.
    """

    def __init__(
        self,
        uri: str,
        *,
        hive_partitioning: bool = True,
        data_format: str | None = None,
        connection_string: str | None = None,
    ) -> None:
        self._uri = uri
        self._scheme = _scheme_of(uri)
        self._format = (data_format or _format_of(uri)).lower()
        self._hive = hive_partitioning
        self._connection_string = connection_string

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection configured for the URI's backend."""
        conn = duckdb.connect()
        if self._scheme in ("s3", "gs", "gcs"):
            conn.execute("INSTALL httpfs; LOAD httpfs;")
            region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
            if region:
                conn.execute(f"SET s3_region = {_q(region)};")
            # Pick up credentials from the standard chain (env, shared config,
            # instance profile). Best-effort: older DuckDB builds lack secrets.
            try:
                conn.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain);")
            except duckdb.Error:
                pass
        elif self._scheme in ("az", "azure", "abfss"):
            conn.execute("INSTALL azure; LOAD azure;")
            # curl transport is the most portable across Linux TLS setups.
            conn.execute("SET azure_transport_option_type = 'curl';")
            ca = self._find_ca_bundle()
            if ca and not os.environ.get("CURL_CA_BUNDLE"):
                os.environ["CURL_CA_BUNDLE"] = ca
            conn_str = self._connection_string or os.environ.get(
                "AZURE_STORAGE_CONNECTION_STRING"
            )
            if conn_str:
                conn.execute(f"SET azure_storage_connection_string = {_q(conn_str)};")
            else:
                account = os.environ.get("AZURE_STORAGE_ACCOUNT")
                if account:
                    conn.execute(f"SET azure_account_name = {_q(account)};")
                conn.execute("SET azure_credential_chain = 'cli;env;managed_identity';")
        return conn

    @staticmethod
    def _find_ca_bundle() -> str | None:
        """Locate a CA bundle for curl-based TLS, falling back to certifi."""
        for path in (
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/ssl/ca-bundle.pem",
        ):
            if Path(path).exists():
                return path
        try:
            import certifi

            return certifi.where()
        except ImportError:
            return None

    def _read_expr(self) -> str:
        """Build the DuckDB table-function expression for the configured URI."""
        if self._format == "csv":
            return f"read_csv_auto('{self._uri}', union_by_name=true)"
        hive = "true" if self._hive else "false"
        return f"read_parquet('{self._uri}', hive_partitioning={hive}, union_by_name=true)"

    def _columns(self, conn: duckdb.DuckDBPyConnection) -> list[str]:
        rows = conn.execute(f"DESCRIBE SELECT * FROM {self._read_expr()} LIMIT 0").fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _resource_column(columns: list[str]) -> str | None:
        if "resource_id" in columns:
            return "resource_id"
        if "resource_hash" in columns:
            return "resource_hash"
        return None

    async def fetch_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        profile: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            where = ["timestamp >= ?", "timestamp < ?"]
            params: list[Any] = [start_time.isoformat(), end_time.isoformat()]
            if profile:
                names = _load_profile_metrics(profile)
                if names:
                    placeholders = ", ".join("?" for _ in names)
                    where.append(f"metric_name IN ({placeholders})")
                    params.extend(names)
            sql = f"SELECT * FROM {self._read_expr()} WHERE {' AND '.join(where)}"
            df = conn.execute(sql, params).df()
        finally:
            conn.close()
        if df.empty:
            return []
        return [normalize_record(r) for r in df.to_dict(orient="records")]

    async def fetch_resources(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rid = self._resource_column(self._columns(conn))
            if rid is None:
                return []
            df = conn.execute(
                f"SELECT DISTINCT {rid} AS resource_id FROM {self._read_expr()} ORDER BY 1"
            ).df()
        finally:
            conn.close()
        return [{"resource_id": row["resource_id"]} for _, row in df.iterrows()]

    async def fetch_metric_names(self) -> list[str]:
        conn = self._connect()
        try:
            df = conn.execute(
                f"SELECT DISTINCT metric_name FROM {self._read_expr()} ORDER BY 1"
            ).df()
        finally:
            conn.close()
        return df["metric_name"].tolist()

    async def fetch_summary(self) -> dict[str, Any]:
        conn = self._connect()
        try:
            rid = self._resource_column(self._columns(conn))
            rid_expr = f"COUNT(DISTINCT {rid})" if rid else "0"
            sql = f"""
                SELECT
                    COUNT(*) AS total_rows,
                    {rid_expr} AS unique_resources,
                    COUNT(DISTINCT metric_name) AS unique_metrics,
                    MIN(timestamp) AS earliest_timestamp,
                    MAX(timestamp) AS latest_timestamp
                FROM {self._read_expr()}
            """
            row = conn.execute(sql).df().iloc[0]
        finally:
            conn.close()
        return {
            "total_rows": int(row["total_rows"]),
            "unique_resources": int(row["unique_resources"]),
            "unique_metrics": int(row["unique_metrics"]),
            "earliest_timestamp": str(row["earliest_timestamp"]),
            "latest_timestamp": str(row["latest_timestamp"]),
        }

    async def get_profile_coverage(self) -> dict[str, Any]:
        """Report feature coverage for every profile over the available data.

        For each profile in ``config/features.yaml``, intersects its expected
        feature names with the distinct ``metric_name`` values present in the
        store. A profile with no expected features is reported as fully covered
        (it requires nothing), which avoids a spurious low-coverage warning.

        Returns:
            ``{"profiles": [{name, description, coverage_percent, available,
            missing, total_expected, total_available}]}``.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT DISTINCT metric_name FROM {self._read_expr()} "
                "WHERE metric_name IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        present = {r[0] for r in rows}

        profiles: list[dict[str, Any]] = []
        for name, meta in _load_all_profiles().items():
            expected = set(meta["features"])
            available = sorted(expected & present)
            missing = sorted(expected - present)
            total_expected = len(expected)
            coverage = (
                100.0 if total_expected == 0 else 100.0 * len(available) / total_expected
            )
            profiles.append(
                {
                    "name": name,
                    "description": meta["description"],
                    "coverage_percent": round(coverage, 1),
                    "available": available,
                    "missing": missing,
                    "total_expected": total_expected,
                    "total_available": len(available),
                }
            )
        return {"profiles": profiles}

    async def get_quality(
        self,
        profile: str | None = None,
        reference_time: datetime | None = None,
        lookback_hours: int | None = None,
        worst_n: int = 20,
    ) -> dict[str, Any]:
        """Report freshness and gap quality per ``(resource, metric)`` series.

        The sample interval is inferred per series as the median positive gap
        between consecutive timestamps, so no cadence is assumed. ``gap_score`` is
        the mean per-series point density (observed over expected points); sparse
        series are listed in ``gaps``. ``freshness`` is measured against
        ``reference_time``, which defaults to the dataset's latest timestamp (not
        wall-clock now) so a static capture reads fresh; a series is "live" when
        its last sample is within ``_FRESH_K`` intervals of the reference, and
        stale series are listed in ``freshness``. ``lag_seconds`` reports
        wall-clock staleness separately. Single-point and constant-timestamp
        series have no inferable interval and are excluded from scoring.

        Args:
            profile: Restrict to this profile's metric names when set.
            reference_time: Freshness reference; defaults to the data's latest timestamp.
            lookback_hours: Restrict analysis to the last N hours before the
                reference; ``None`` (the default) assesses the full window.
            worst_n: Cap on the number of series listed in ``freshness`` and ``gaps``.

        Returns:
            ``{"summary": {...}, "freshness": [...], "gaps": [...]}``.
        """
        names = _load_profile_metrics(profile) if profile else []
        name_filter = " AND list_contains($names, metric_name)" if names else ""

        conn = self._connect()
        try:
            # Pin UTC so TIMESTAMPTZ->TIMESTAMP casts (and lag vs wall-clock now)
            # are computed in UTC regardless of the host timezone.
            conn.execute("SET TimeZone='UTC';")

            rid = self._resource_column(self._columns(conn))
            rid_expr = rid if rid else "CAST(NULL AS VARCHAR)"

            bounds_params: dict[str, Any] = {"ref": reference_time}
            if names:
                bounds_params["names"] = names
            bounds = conn.execute(
                f"SELECT COALESCE($ref::TIMESTAMP, MAX(timestamp)::TIMESTAMP) AS ref_ts, "
                f"MAX(timestamp)::TIMESTAMP AS data_max_ts "
                f"FROM {self._read_expr()} "
                f"WHERE timestamp IS NOT NULL{name_filter}",
                bounds_params,
            ).fetchone()
            ref_ts, data_max_ts = bounds[0], bounds[1]
            if data_max_ts is None:
                return {
                    "summary": {
                        "freshness_score": 0.0,
                        "gap_score": 0.0,
                        "lag_seconds": None,
                        "total_series": 0,
                        "series_analyzed": 0,
                        "reference_time": None,
                    },
                    "freshness": [],
                    "gaps": [],
                }

            q_params: dict[str, Any] = {"ref": ref_ts, "k": _FRESH_K}
            lower_bound = ""
            if lookback_hours is not None:
                lower_bound = "AND ts > $ref::TIMESTAMP - ($lookback * INTERVAL '1 hour')"
                q_params["lookback"] = lookback_hours
            if names:
                q_params["names"] = names

            series_sql = f"""
                WITH src AS (
                    SELECT {rid_expr} AS resource_id, metric_name, timestamp::TIMESTAMP AS ts
                    FROM {self._read_expr()}
                    WHERE timestamp IS NOT NULL{name_filter}
                ),
                windowed AS (
                    SELECT resource_id, metric_name, ts
                    FROM src
                    WHERE ts <= $ref::TIMESTAMP {lower_bound}
                ),
                deltas AS (
                    SELECT resource_id, metric_name, ts,
                        epoch(ts - LAG(ts) OVER (
                            PARTITION BY resource_id, metric_name ORDER BY ts)) AS dsec
                    FROM windowed
                ),
                series AS (
                    SELECT resource_id, metric_name,
                        COUNT(*) AS pts,
                        MIN(ts) AS first_ts,
                        MAX(ts) AS last_ts,
                        quantile_cont(dsec, 0.5) FILTER (WHERE dsec > 0) AS interval_s,
                        max(dsec) FILTER (WHERE dsec > 0) AS max_delta_s
                    FROM deltas
                    GROUP BY resource_id, metric_name
                )
                SELECT resource_id, metric_name, pts, last_ts, interval_s,
                    CASE WHEN interval_s IS NULL THEN NULL
                         ELSE least(1.0, pts::DOUBLE /
                             (floor(epoch(last_ts - first_ts) / interval_s) + 1))
                    END AS density,
                    CASE WHEN interval_s IS NULL THEN NULL
                         ELSE epoch($ref::TIMESTAMP - last_ts) / interval_s
                    END AS intervals_behind,
                    CASE WHEN interval_s IS NULL THEN NULL
                         ELSE max_delta_s / interval_s
                    END AS longest_gap_intervals,
                    CASE WHEN interval_s IS NULL THEN NULL
                         WHEN epoch($ref::TIMESTAMP - last_ts) / interval_s <= $k THEN TRUE
                         ELSE FALSE
                    END AS is_live
                FROM series
                ORDER BY resource_id, metric_name
            """
            df = conn.execute(series_sql, q_params).df()
        finally:
            conn.close()

        total_series = len(df)
        analyzed = df[df["interval_s"].notna()]
        series_analyzed = len(analyzed)

        if series_analyzed == 0:
            freshness_score = 0.0
            gap_score = 0.0
        else:
            gap_score = round(100.0 * float(analyzed["density"].mean()), 1)
            freshness_score = round(
                100.0 * float(analyzed["is_live"].astype(bool).mean()), 1
            )

        dead = (
            analyzed[~analyzed["is_live"].astype(bool)]
            .sort_values("intervals_behind", ascending=False)
            .head(worst_n)
        )
        freshness = [
            {
                "resource_id": r["resource_id"],
                "metric_name": r["metric_name"],
                "last_timestamp": str(r["last_ts"]),
                "intervals_behind": round(float(r["intervals_behind"]), 1),
            }
            for _, r in dead.iterrows()
        ]

        gappy = analyzed[analyzed["density"] < 1.0].sort_values("density").head(worst_n)
        gaps = [
            {
                "resource_id": r["resource_id"],
                "metric_name": r["metric_name"],
                "density": round(float(r["density"]), 3),
                "missing_pct": round(100.0 * (1.0 - float(r["density"])), 1),
                "longest_gap_intervals": round(float(r["longest_gap_intervals"]), 1),
            }
            for _, r in gappy.iterrows()
        ]

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        lag_seconds = round((now - data_max_ts).total_seconds(), 1)

        return {
            "summary": {
                "freshness_score": freshness_score,
                "gap_score": gap_score,
                "lag_seconds": lag_seconds,
                "total_series": total_series,
                "series_analyzed": series_analyzed,
                "reference_time": str(ref_ts),
            },
            "freshness": freshness,
            "gaps": gaps,
        }
