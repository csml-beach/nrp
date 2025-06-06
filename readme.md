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

## Notes
- Ensure your user has proper RBAC access to your namespace
- The container runs as root to allow writing to mounted volumes
- You can change the script being executed by modifying the `SCRIPT_NAME` env var in the YAML
