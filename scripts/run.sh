#!/usr/bin/env bash
# Run a command inside the flexor Docker container.
#   scripts/run.sh python train.py --config configs/xxx.yaml
#   scripts/run.sh            # interactive shell
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export FLEXOR_DATA_DIR="${FLEXOR_DATA_DIR:-/home/sgsuh/data/torchvision}"

# Create the data dir as the host user so Docker does not create it as root.
mkdir -p "${FLEXOR_DATA_DIR}"

cd "${REPO_ROOT}"
if [ "$#" -eq 0 ]; then
  exec docker compose run --rm flexor bash
fi
exec docker compose run --rm flexor "$@"
