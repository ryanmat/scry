# Scry - LogicModules

This directory contains LogicModule components for integrating Scry with LogicMonitor.

---

## Contents

| File | Type | Status |
|------|------|--------|
| `REMEDIATION_DESIGN.md` | Design Document | Future implementation |
| `external_alert_handler.ps1` | External Alerting | Workaround available |

---

## External Alerting Workaround (external_alert_handler.ps1)

Interim solution until RemediationSource is available.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SCRY_API_URL` | Prediction API endpoint | Prediction API URL |
| `SCRY_LOG_PATH` | Script directory | Log file location |
| `SCRY_DRY_RUN` | false | Enable dry-run mode |

### Installation

1. Copy script to `<collector>/agent/local/bin/`
2. Configure External Alerting in LM portal
3. Set script path as alert destination
4. Create alert rule to trigger on K8s alerts

---

## RemediationSource (Future)

See `REMEDIATION_DESIGN.md` for full implementation design.

RemediationSource will be available in Q1 2026.

---

## API Reference

Prediction API: `https://your-scry-api.example.com`

### Endpoints

- `GET /health` - Health check
- `GET /predict?resource_id=<id>` - Get prediction for resource
- `POST /predict` - Batch prediction with metrics

### Response Format

```json
{
    "resource_id": "pod-name",
    "cluster_id": 2,
    "cluster_name": "PRE_FAILURE",
    "confidence": 0.89,
    "action": "DIAGNOSTIC",
    "priority": "HIGH"
}
```
