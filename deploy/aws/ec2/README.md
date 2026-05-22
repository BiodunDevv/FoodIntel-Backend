# FoodIntel Backend: EC2 Deployment With PEM

Use this path when you already have an EC2 instance and a `.pem` SSH key.

## What this script does

`scripts/deploy_ec2.sh` will:

1. SSH into the EC2 instance
2. Create `/home/ubuntu/foodintel-backend` by default
3. `rsync` the backend code
4. Upload the local backend `.env`
5. Create a Python virtualenv
6. Install `requirements-api.txt`
7. Start `uvicorn app.main:app --host 0.0.0.0 --port 8000`
8. Verify `/health`

## Required on the EC2 instance

- Ubuntu or similar Linux host
- `python3`, `python3-venv`, `curl`, `rsync`
- Security group inbound rule allowing:
  - `22` from your IP
  - `8000` from your IP for a fast direct test

Later, for a proper public API:

- Put Nginx in front of the app
- Add HTTPS with a domain and certificate
- Restrict direct `8000` access

## Run it

From `foodintel-backend/`:

```bash
chmod +x scripts/deploy_ec2.sh

EC2_HOST=ec2-12-34-56-78.compute-1.amazonaws.com \
EC2_USER=ubuntu \
PEM_PATH=../FoodIntel-Backend.pem \
./scripts/deploy_ec2.sh
```

If the host uses a different username:

- Amazon Linux: `ec2-user`
- Ubuntu: `ubuntu`

## Result

After a successful run:

- Health: `http://YOUR_EC2_HOST:8000/health`
- Docs: `http://YOUR_EC2_HOST:8000/docs`

## Important production note

This direct EC2 path is fast and workable, but ECS/Fargate remains the cleaner long-term production setup because it gives:

- repeatable container deploys
- health-managed restarts
- load balancer integration
- easier scaling

If you want, we can still use EC2 today to get you live quickly, then move to ECS cleanly after.
