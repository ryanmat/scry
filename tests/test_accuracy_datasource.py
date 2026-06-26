# Description: Tests for Scry_Accuracy.xml DataSource structure.
# Description: Validates XML parsing, datapoint names, and namevalue post-processing pattern.

"""Tests for Scry_Accuracy DataSource XML structure."""

import xml.etree.ElementTree as ET
from pathlib import Path

XML_PATH = Path(__file__).parent.parent / "logicmodules" / "datasources" / "Scry_Accuracy.xml"

# All flat keys from the /accuracy API response
EXPECTED_DATAPOINTS = [
    "Picp15m",
    "Picp1h",
    "Picp4h",
    "Picp24h",
    "Mae15m",
    "Mae1h",
    "Mae4h",
    "Mae24h",
    "Mase15m",
    "Mase1h",
    "Mase4h",
    "Mase24h",
    "Mpiw15m",
    "Mpiw1h",
    "Mpiw4h",
    "Mpiw24h",
    "TransitionRate",
    "ConfidenceStd",
    "DominantClusterPct",
    "ObservationCount",
    "ApiStatus",
    "ApiLatencyMs",
]


class TestAccuracyXmlStructure:
    """Tests for Scry_Accuracy.xml DataSource structure."""

    def test_accuracy_xml_valid(self):
        """XML parses correctly without errors."""
        tree = ET.parse(XML_PATH)
        root = tree.getroot()
        assert root.tag == "feed"

    def test_accuracy_xml_datapoints_match_api(self):
        """All flat keys from /accuracy response have matching datapoints."""
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        param_fields = []
        for dp in root.iter("dataPoint"):
            post_param = dp.find("postProcessorParam")
            if post_param is not None and post_param.text:
                param_fields.append(post_param.text)

        for field in EXPECTED_DATAPOINTS:
            assert field in param_fields, f"Missing datapoint for API key: {field}"

        assert len(param_fields) == len(EXPECTED_DATAPOINTS), (
            f"Expected {len(EXPECTED_DATAPOINTS)} datapoints, got {len(param_fields)}"
        )

    def test_accuracy_xml_namevalue_pattern(self):
        """All datapoints use rawDataFieldName=output and postProcessorMethod=namevalue."""
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        for dp in root.iter("dataPoint"):
            name = dp.find("name").text
            raw_field = dp.find("rawDataFieldName")
            post_method = dp.find("postProcessorMethod")

            assert raw_field is not None and raw_field.text == "output", (
                f"Datapoint {name}: rawDataFieldName must be 'output', got '{raw_field.text if raw_field is not None else None}'"
            )
            assert post_method is not None and post_method.text == "namevalue", (
                f"Datapoint {name}: postProcessorMethod must be 'namevalue', got '{post_method.text if post_method is not None else None}'"
            )

    def test_accuracy_xml_collect_interval(self):
        """Collect interval is 15 minutes."""
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        interval = root.find(".//collectInterval")
        assert interval is not None
        assert interval.text == "15"

    def test_accuracy_xml_alert_expressions(self):
        """Key datapoints have appropriate alert expressions."""
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        alert_map = {}
        for dp in root.iter("dataPoint"):
            name = dp.find("name").text
            alert_expr = dp.find("alertExpr")
            if alert_expr is not None and alert_expr.text:
                alert_map[name] = alert_expr.text

        # PICP thresholds (lower is worse)
        assert "Picp15m" in alert_map
        assert "Picp1h" in alert_map

        # MASE thresholds (higher is worse)
        assert "Mase15m" in alert_map

        # Stability thresholds
        assert "TransitionRate" in alert_map
        assert "ConfidenceStd" in alert_map

        # Operational
        assert "ApiStatus" in alert_map
        assert "ApiLatencyMs" in alert_map
