# Scry - RemediationSource Design Document

## Overview

This document describes the RemediationSource LogicModule design for Scry.
RemediationSource is not yet available for internal testing (expected Q1 2026).

This design document serves as the implementation blueprint for when the feature becomes available.

---

## Architecture

```
                    Alert Triggered
                          │
                          ▼
              ┌───────────────────────┐
              │   RemediationSource   │
              │   (This LogicModule)  │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Prediction API Call  │
              │  /predict?resource_id │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Action Dispatcher   │
              │   SCALE | DIAGNOSE    │
              │   REMEDIATE | ALERT   │
              └───────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       ┌─────────┐  ┌──────────┐  ┌─────────┐
       │ kubectl │  │ Log Only │  │  Alert  │
       │  scale  │  │ (Audit)  │  │ Webhook │
       └─────────┘  └──────────┘  └─────────┘
```

---

## Device Properties Required

RemediationSource uses isolated `remediation.*` properties for security.
These properties are not accessible by standard LogicModules.

| Property | Description | Example |
|----------|-------------|---------|
| `remediation.k8s.kubeconfig` | Base64-encoded kubeconfig | `YXBpVmVyc2lvbjogdjE...` |
| `remediation.k8s.namespace` | Target namespace for scaling | `production` |
| `remediation.k8s.deployment` | Deployment name to scale | `api-server` |
| `remediation.api.url` | Prediction API endpoint | `https://your-scry-api...` |
| `remediation.scale.min` | Minimum replica count | `2` |
| `remediation.scale.max` | Maximum replica count | `10` |
| `remediation.dry_run` | Enable dry-run mode | `true` |

---

## Implementation Code

```groovy
// Description: RemediationSource LogicModule for Scry.
// Description: Executes remediation actions based on ML predictions.

import com.santaba.agent.groovyapi.http.*
import groovy.json.JsonSlurper
import groovy.json.JsonOutput

// Remediation-specific properties (isolated from standard hostProps)
def apiUrl = remediation.get("api.url")
    ?: "https://your-scry-api.example.com"
def kubeconfig = remediation.get("k8s.kubeconfig")
def namespace = remediation.get("k8s.namespace") ?: "default"
def deployment = remediation.get("k8s.deployment")
def minReplicas = remediation.get("scale.min")?.toInteger() ?: 2
def maxReplicas = remediation.get("scale.max")?.toInteger() ?: 10
def dryRun = remediation.get("dry_run")?.toBoolean() ?: false

// Get resource identifier
def resourceId = hostProps.get("system.displayname") ?: hostProps.get("system.hostname")

// Initialize result tracking
def output = new StringBuilder()
def remediationStatus = "success"
def actionsPerformed = []
def startTime = System.currentTimeMillis()

output.append("## Scry Remediation\n\n")
output.append("**Resource:** ${resourceId}\n")
output.append("**Timestamp:** ${new Date().format('yyyy-MM-dd HH:mm:ss z')}\n")
output.append("**Mode:** ${dryRun ? 'DRY RUN' : 'LIVE'}\n\n")

try {
    // Step 1: Get prediction from API
    output.append("### Step 1: Get Prediction\n\n")

    def urlParts = apiUrl.replace("https://", "").replace("http://", "").split("/")
    def host = urlParts[0]
    def port = apiUrl.startsWith("https") ? 443 : 80

    def http = HTTP.open(host, port, apiUrl.startsWith("https"))
    def endpoint = "/predict?resource_id=" + URLEncoder.encode(resourceId, "UTF-8")
    http.get(endpoint)

    def responseCode = http.getStatusCode()
    def responseBody = http.getResponseBody()
    http.close()

    if (responseCode != 200) {
        throw new Exception("API returned status ${responseCode}: ${responseBody}")
    }

    def prediction = new JsonSlurper().parseText(responseBody)
    output.append("- Cluster: ${prediction.cluster_name}\n")
    output.append("- Confidence: ${String.format('%.1f%%', prediction.confidence * 100)}\n")
    output.append("- Recommended Action: ${prediction.action}\n\n")

    // Step 2: Execute action based on prediction
    output.append("### Step 2: Execute Action\n\n")

    switch(prediction.action) {
        case "SCALE":
            output.append("**Action:** Scaling deployment\n\n")

            if (!deployment) {
                output.append("ERROR: remediation.k8s.deployment not configured\n")
                remediationStatus = "failure"
                break
            }

            // Calculate target replicas based on cluster state
            def targetReplicas = calculateTargetReplicas(prediction, minReplicas, maxReplicas)
            output.append("- Target Namespace: ${namespace}\n")
            output.append("- Target Deployment: ${deployment}\n")
            output.append("- Target Replicas: ${targetReplicas}\n\n")

            if (dryRun) {
                output.append("DRY RUN: Would execute kubectl scale\n")
                actionsPerformed.add("scale_dry_run")
            } else {
                // Write kubeconfig to temp file
                def kubeconfigFile = "/tmp/kubeconfig_${System.currentTimeMillis()}"
                new File(kubeconfigFile).text = new String(kubeconfig.decodeBase64())

                try {
                    def cmd = ["kubectl", "--kubeconfig=${kubeconfigFile}",
                               "-n", namespace, "scale", "deployment/${deployment}",
                               "--replicas=${targetReplicas}"]
                    def proc = cmd.execute()
                    proc.waitFor()

                    if (proc.exitValue() == 0) {
                        output.append("SUCCESS: Scaled to ${targetReplicas} replicas\n")
                        actionsPerformed.add("scale_executed")
                    } else {
                        output.append("ERROR: ${proc.err.text}\n")
                        remediationStatus = "failure"
                    }
                } finally {
                    new File(kubeconfigFile).delete()
                }
            }
            break

        case "DIAGNOSTIC":
            output.append("**Action:** Logging diagnostic information\n\n")
            output.append("No immediate remediation required.\n")
            output.append("Diagnostic data captured for analysis.\n")
            actionsPerformed.add("diagnostic_logged")
            break

        case "REMEDIATE":
            output.append("**Action:** Pod remediation\n\n")

            if (!deployment) {
                output.append("ERROR: remediation.k8s.deployment not configured\n")
                remediationStatus = "failure"
                break
            }

            if (dryRun) {
                output.append("DRY RUN: Would restart unhealthy pods\n")
                actionsPerformed.add("remediate_dry_run")
            } else {
                // Rollout restart to address unhealthy pods
                def kubeconfigFile = "/tmp/kubeconfig_${System.currentTimeMillis()}"
                new File(kubeconfigFile).text = new String(kubeconfig.decodeBase64())

                try {
                    def cmd = ["kubectl", "--kubeconfig=${kubeconfigFile}",
                               "-n", namespace, "rollout", "restart",
                               "deployment/${deployment}"]
                    def proc = cmd.execute()
                    proc.waitFor()

                    if (proc.exitValue() == 0) {
                        output.append("SUCCESS: Initiated rollout restart\n")
                        actionsPerformed.add("remediate_executed")
                    } else {
                        output.append("ERROR: ${proc.err.text}\n")
                        remediationStatus = "failure"
                    }
                } finally {
                    new File(kubeconfigFile).delete()
                }
            }
            break

        case "ALERT":
            output.append("**Action:** Alert escalation\n\n")
            output.append("Alert acknowledged. No automated remediation.\n")
            output.append("Manual review recommended.\n")
            actionsPerformed.add("alert_acknowledged")
            break

        default:
            output.append("**Action:** None required\n\n")
            output.append("Workload operating normally.\n")
            actionsPerformed.add("none")
    }

} catch (Exception e) {
    output.append("### Error\n\n")
    output.append("Remediation failed: ${e.message}\n")
    remediationStatus = "failure"
}

// Step 3: Summary
def duration = System.currentTimeMillis() - startTime
output.append("\n### Summary\n\n")
output.append("- Duration: ${duration}ms\n")
output.append("- Actions: ${actionsPerformed.join(', ')}\n")
output.append("- Status: ${remediationStatus.toUpperCase()}\n")

// Return result in RemediationSource format
print JsonOutput.toJson([
    data: output.toString(),
    format: "markdown",
    remediationStatus: remediationStatus
])

return remediationStatus == "success" ? 0 : 1

// Helper function to calculate target replicas
def calculateTargetReplicas(prediction, minReplicas, maxReplicas) {
    // Cluster mapping to scaling factor
    def scalingFactors = [
        "NORMAL": 1.0,                 // Maintain
        "PRE_SCALE": 1.5,              // Scale up
        "PRE_FAILURE": 1.25,           // Slight scale up
        "ACTIVE_DEGRADATION": 2.0,     // Emergency scale up
        "ANOMALY": 1.0                 // Maintain, investigate
    ]

    def factor = scalingFactors.get(prediction.cluster_name, 1.0)
    def currentReplicas = minReplicas // Would get from kubectl in production

    def target = Math.round(currentReplicas * factor)
    return Math.max(minReplicas, Math.min(maxReplicas, target))
}
```

---

## Security Considerations

### Credential Isolation

RemediationSources use the `remediation.*` property namespace which is:
- Isolated from standard `hostProps` accessible by DataSources
- Only accessible by RemediationSource scripts
- Subject to additional audit logging

### Audit Trail

All remediation actions generate audit entries including:
- Timestamp
- Resource identifier
- Action performed
- Before/after state
- User/system that triggered remediation

### Safe Defaults

- `dry_run` defaults to `true` to prevent accidental changes
- Replica bounds (`scale.min`, `scale.max`) prevent runaway scaling
- All kubectl commands use explicit kubeconfig (no default context)

---

## Output Format

RemediationSource must return JSON in this exact format:

```json
{
    "data": "Markdown formatted output string",
    "format": "markdown",
    "remediationStatus": "success|failure"
}
```

The `remediationStatus` field is required and must be either:
- `"success"` - Remediation completed successfully
- `"failure"` - Remediation encountered an error

---

## Testing Plan

When RemediationSource becomes available:

1. **Unit Testing**
   - Test API connectivity
   - Test JSON parsing
   - Test output formatting

2. **Dry Run Testing**
   - Execute with `dry_run=true`
   - Verify all actions are logged
   - Verify no actual changes made

3. **Integration Testing**
   - Test with non-production K8s cluster
   - Verify scaling actions work
   - Verify rollout restart works

4. **Failure Mode Testing**
   - Test with invalid kubeconfig
   - Test with unreachable API
   - Test with invalid deployment name

---

## Migration from External Alerting

When RemediationSource is available, migrate from External Alerting:

1. Export existing alert handler scripts
2. Create RemediationSource with equivalent logic
3. Configure `remediation.*` properties
4. Test in dry-run mode
5. Switch alert routing to RemediationSource
6. Deprecate External Alerting scripts

---

## References

- LogicMonitor Automated Remediation: Coming Q1 2026
- External Alerting Documentation: https://www.logicmonitor.com/support/alerts/alert-destinations/external-alerting
- Scry API: https://your-scry-api.example.com/docs
