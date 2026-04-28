This repository contains Kubernetes job definitions and Python scripts for running GPU-enabled tests and experiments on the NRP Nautilus cluster.

## Structure

- `jobs/` — YAML job templates to submit to Kubernetes
- `scripts/` — Python scripts to be run inside the job containers

## How to Use

1. Clone this repo:
   ```bash
   git clone https://github.com/csml-beach/nrp.git
   cd nrp
   ```

2. Submit a job:
   ```bash
   kubectl apply -f jobs/gpu-test-git.yaml -n csml-beach
   ```

3. Monitor the job:
   ```bash
   kubectl get pods -n csml-beach
   kubectl logs <pod-name> -n csml-beach
   ```

4. Check results:
   Output is saved to the PVC mounted at `/mnt/data`, e.g.:
   ```
   /mnt/data/output_gpu_test.py.txt
   ```

## PyTorch & Data Science Environment

We have set up a ready-to-use PyTorch and Data Science environment utilizing the NRP scientific images.

- **Job Template:** `jobs/pytorch-gpu-run.yaml`
- **Image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Docker Hub)
- **Test Script:** `scripts/pytorch-test.py`

### How to use:
1. Ensure your script is in the `scripts/` directory and pushed to GitHub.
2. Update the `SCRIPT_NAME` env var in `jobs/pytorch-gpu-run.yaml` if needed.
3. Submit the job:
   ```bash
   kubectl apply -f jobs/pytorch-gpu-run.yaml -n csml-beach
   ```
4. Check the results in `/mnt/data/output_pytorch-test.py.txt` via the `debug-shell`.

## Troubleshooting & Tips

### Authentication (OIDC)
Nautilus uses OIDC for authentication. If you get an `invalid_grant` error:
1. Refresh your config from the [Nautilus Portal](https://portal.nrp-nautilus.io/).
2. If you need to switch identities (e.g., from ORCID to CSULB), clear your sessions:
   - [Authentik Logout](https://authentik.nrp-nautilus.io/flows/-/default/invalidation/)
   - [CILogon Logout](https://cilogon.org/logout)
3. Ensure you have the `kubelogin` plugin installed: `brew install int128/kubelogin/kubelogin`.

### Resource Quotas & Ratios
Nautilus enforces strict CPU/Memory limit-to-request ratios (usually 1:1 or up to 1.2).
- **Tip:** Set `requests` equal to `limits` to ensure your pod is scheduled without being blocked by admission controllers.
- Example:
  ```yaml
  resources:
    limits:
      cpu: "1"
      memory: "2Gi"
    requests:
      cpu: "1"
      memory: "2Gi"
  ```

### PVC Multi-Attach Errors
If a new job is stuck in `ContainerCreating` with a `Multi-Attach error`, it means a previous pod is still holding the volume on another node.
1. Check for terminating pods: `kubectl get pods -n csml-beach`
2. Force delete the stuck pod: 
   ```bash
   kubectl delete pod <pod-name> -n csml-beach --force --grace-period=0
   ```

### Large Images
The default PRP container image (`gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp`) is ~16GB. It is normal for the pod to stay in `ContainerCreating` for several minutes while the image is pulled to a new node.
