#!/usr/bin/env bash
# Provision an A100 80GB VM on GCP and prepare the experiment environment.
# Run once from your local machine. Adjust ZONE if quota is elsewhere.

set -euo pipefail

PROJECT="emg2qwerty-team-shared"
ZONE="us-central1-a"
VM="vla-train"
MACHINE="a2-highgpu-1g"   # 1× A100 40GB, 12 vCPUs, 85 GB RAM (LoRA fits comfortably)
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=== Creating VM $VM in $PROJECT ($ZONE) ==="
gcloud compute instances create "$VM" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --accelerator="type=nvidia-tesla-a100,count=1" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size=200GB \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True"

echo ""
echo "=== Cloning repo and installing dependencies ==="
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" -- bash -s <<'REMOTE'
set -euo pipefail
git clone https://github.com/sttawm/vla-grpo-experiments.git ~/experiments
cd ~/experiments
pip install -r requirements.txt
# OpenVLA-OFT from source (not on PyPI)
pip install git+https://github.com/openvla/openvla.git
echo "Setup complete."
REMOTE

echo ""
echo "VM ready. SSH with:"
echo "  gcloud compute ssh $VM --zone=$ZONE --project=$PROJECT"
echo ""
echo "Then run:"
echo "  cd experiments"
echo "  python probe_schema.py        # verify Bridge field names first"
echo "  python baseline_eval.py       # ~200 episodes, ~3 hr — run in tmux"
echo ""
echo "After baseline completes:"
echo "  python grpo_train.py --steps 1000"
echo ""
echo "Note: spot instance may be preempted. Both scripts are resumable."
echo "If preempted, restart the instance and re-run — results are saved incrementally."
