#!/usr/bin/env bash
# SSH tunnel for remote TensorBoard → http://localhost:6006
# Usage: SSH_HOST=your.cluster.example ./scripts/tensorboard_tunnel.sh
set -euo pipefail
: "${SSH_HOST:?Set SSH_HOST to your cluster login host (or SSH config alias)}"
echo "TensorBoard tunnel: http://localhost:6006  (via ${SSH_HOST})"
echo "Press Ctrl+C to stop."
exec ssh -N -L 6006:127.0.0.1:6006 "${SSH_HOST}"
