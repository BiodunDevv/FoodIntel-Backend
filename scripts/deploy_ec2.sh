#!/usr/bin/env bash
set -euo pipefail

EC2_HOST="${EC2_HOST:-}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_PORT="${EC2_PORT:-22}"
PEM_PATH="${PEM_PATH:-}"
REMOTE_DIR="${REMOTE_DIR:-/home/${EC2_USER}/foodintel-backend}"
REMOTE_ENV_PATH="${REMOTE_ENV_PATH:-${REMOTE_DIR}/.env}"
LOCAL_ENV_PATH="${LOCAL_ENV_PATH:-.env}"

if [[ -z "$EC2_HOST" || -z "$PEM_PATH" ]]; then
  echo "Set EC2_HOST and PEM_PATH before running this script." >&2
  echo "Example:" >&2
  echo "  EC2_HOST=ec2-xx-xx-xx-xx.compute-1.amazonaws.com PEM_PATH=../FoodIntel-Backend.pem ./scripts/deploy_ec2.sh" >&2
  exit 1
fi

if [[ ! -f "$PEM_PATH" ]]; then
  echo "PEM file not found: $PEM_PATH" >&2
  exit 1
fi

chmod 400 "$PEM_PATH"

SSH_OPTS=(
  -i "$PEM_PATH"
  -p "$EC2_PORT"
  -o StrictHostKeyChecking=accept-new
)

SCP_OPTS=(
  -i "$PEM_PATH"
  -P "$EC2_PORT"
  -o StrictHostKeyChecking=accept-new
)

echo "Creating remote directory..."
ssh "${SSH_OPTS[@]}" "${EC2_USER}@${EC2_HOST}" "mkdir -p '${REMOTE_DIR}' '${REMOTE_DIR}/uploads' '${REMOTE_DIR}/logs'"

echo "Syncing backend files..."
rsync -avz --delete \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude "ml/data" \
  --exclude "ml/dataset_master" \
  --exclude "ml/dataset_live" \
  --exclude "ml/dataset_feedback" \
  --exclude "ml/dataset_feedback_live" \
  --exclude "ml/raw_sources" \
  --exclude "ml/reports" \
  --exclude "uploads/*" \
  ./ "${EC2_USER}@${EC2_HOST}:${REMOTE_DIR}/"

if [[ -f "$LOCAL_ENV_PATH" ]]; then
  echo "Uploading environment file..."
  scp "${SCP_OPTS[@]}" "$LOCAL_ENV_PATH" "${EC2_USER}@${EC2_HOST}:${REMOTE_ENV_PATH}"
fi

echo "Installing runtime dependencies and restarting API..."
ssh "${SSH_OPTS[@]}" "${EC2_USER}@${EC2_HOST}" <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"
mkdir -p logs uploads
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements-api.txt
pkill -f "uvicorn app.main:app" || true
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 > logs/uvicorn.log 2>&1 &
sleep 3
curl -fsS http://127.0.0.1:8000/health >/tmp/foodintel-health.json
cat /tmp/foodintel-health.json
EOF

echo
echo "Backend deployed."
echo "API health URL: http://${EC2_HOST}:8000/health"
echo "API docs URL:   http://${EC2_HOST}:8000/docs"
