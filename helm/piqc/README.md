# piqc Helm chart

Installs the same read-only RBAC as [`deploy/rbac.yaml`](../../deploy/rbac.yaml), plus a CronJob that runs `piqc scan` on a schedule and pushes results to the Paralleliq platform — one command instead of three, and no manual YAML editing.

Commands below assume you've cloned this repo and are running them from its root (`git clone https://github.com/paralleliq/piqc && cd piqc`).

## Before you install: read what this applies

```bash
helm template my-piqc ./helm/piqc \
  --set clusterId=<your-cluster-id> \
  --set pushUrl=https://api.paralleliq.ai \
  --set apiKey=<your-api-key>
```

This prints every manifest the chart would create — a ServiceAccount, a ClusterRole (read-only, `get`/`list` only, see the comment in `templates/clusterrole.yaml` for the one `pods/exec` `create` verb and exactly what it's for), a ClusterRoleBinding, a Secret, and a CronJob. Nothing here writes to, patches, or deletes anything in your cluster.

## Install

1. Register this cluster with the platform (dashboard's Add Cluster flow, or `POST /v1/clusters`) to get a `clusterId` and `apiKey`. The key is shown once — save it.
2. Either let the chart create a Secret from that key (quick start), or create one yourself first and point `apiKeySecret.name` at it (recommended — keeps the real key out of `helm history`):
   ```bash
   kubectl create secret generic piqc-api-key --from-literal=apiKey=<your-api-key>
   ```
3. Install:
   ```bash
   helm install my-piqc ./helm/piqc \
     --set clusterId=<your-cluster-id> \
     --set pushUrl=https://api.paralleliq.ai \
     --set apiKeySecret.name=piqc-api-key
   ```

## Values

See [`values.yaml`](values.yaml) — every field has an inline comment. The two you must set are `clusterId` and `pushUrl`; `apiKey` or `apiKeySecret.name` is required one way or the other.

Want hourly instead of daily? `--set schedule="0 * * * *"` — no other change needed.
