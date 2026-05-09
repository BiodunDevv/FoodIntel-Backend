# FoodIntel Backend: AWS ECS/Fargate Deployment

This backend is designed to deploy as a Docker container on ECS Fargate behind an Application Load Balancer.

## Recommended Production Shape

- **Runtime:** ECS Fargate service
- **Image registry:** ECR
- **Database:** MongoDB Atlas
- **File uploads:** Cloudinary for production uploads
- **Secrets:** AWS Systems Manager Parameter Store
- **Logs:** CloudWatch Logs
- **Ingress:** Application Load Balancer with HTTPS certificate from ACM
- **Container port:** `8000`
- **Health check:** `/health`

## 1. Local Sanity Check

From `foodintel-backend/`:

```bash
docker build -t foodintel-api .
docker run --rm -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e MONGODB_URI="YOUR_MONGODB_ATLAS_URI" \
  -e MONGODB_DB_NAME="foodintel_db" \
  -e JWT_SECRET_KEY="replace-with-a-long-random-secret" \
  -e CORS_ORIGINS="http://localhost:3000" \
  -e RETRAIN_ON_FEEDBACK=false \
  foodintel-api
```

Check:

```bash
curl http://localhost:8000/health
```

## 2. Push Image To ECR

From `foodintel-backend/`:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPOSITORY=foodintel-api
export IMAGE_TAG=$(git rev-parse --short HEAD)

./scripts/build_push_ecr.sh
```

The script prints the final ECR image URI. Use that URI in the ECS task definition.

## 3. Store Secrets In Parameter Store

```bash
aws ssm put-parameter \
  --name /foodintel/prod/MONGODB_URI \
  --type SecureString \
  --value "YOUR_MONGODB_ATLAS_URI" \
  --region "$AWS_REGION"

aws ssm put-parameter \
  --name /foodintel/prod/JWT_SECRET_KEY \
  --type SecureString \
  --value "replace-with-a-long-random-secret" \
  --region "$AWS_REGION"

aws ssm put-parameter \
  --name /foodintel/prod/CLOUDINARY_CLOUD_NAME \
  --type SecureString \
  --value "YOUR_CLOUDINARY_CLOUD_NAME" \
  --region "$AWS_REGION"

aws ssm put-parameter \
  --name /foodintel/prod/CLOUDINARY_API_KEY \
  --type SecureString \
  --value "YOUR_CLOUDINARY_API_KEY" \
  --region "$AWS_REGION"

aws ssm put-parameter \
  --name /foodintel/prod/CLOUDINARY_API_SECRET \
  --type SecureString \
  --value "YOUR_CLOUDINARY_API_SECRET" \
  --region "$AWS_REGION"
```

## 4. Prepare ECS IAM Roles

Make sure the task execution role can:

- Pull from ECR
- Write CloudWatch logs
- Read SSM parameters under `/foodintel/prod/*`

The default `ecsTaskExecutionRole` often already has ECR and CloudWatch permissions. Add SSM read access:

```json
{
  "Effect": "Allow",
  "Action": [
    "ssm:GetParameters",
    "ssm:GetParameter",
    "kms:Decrypt"
  ],
  "Resource": [
    "arn:aws:ssm:REGION:ACCOUNT_ID:parameter/foodintel/prod/*"
  ]
}
```

## 5. Register Task Definition

Copy `deploy/aws/ecs-task-definition.template.json` to a local ignored file and replace:

- `<AWS_ACCOUNT_ID>`
- `<AWS_REGION>`
- `<IMAGE_TAG>`
- `https://your-frontend-domain.com`

Then register:

```bash
aws logs create-log-group \
  --log-group-name /ecs/foodintel-api \
  --region "$AWS_REGION" || true

aws ecs register-task-definition \
  --cli-input-json file://deploy/aws/ecs-task-definition.json \
  --region "$AWS_REGION"
```

## 6. Create ECS Service

Use the AWS console for the first service setup because VPC, subnet, security group, and ALB wiring are easier to verify visually.

Recommended choices:

- Launch type: **Fargate**
- Desired tasks: `1` to start
- CPU/memory: `1 vCPU / 2 GB`
- Public IP: enabled if using public subnets
- Load balancer: Application Load Balancer
- Listener: `HTTPS 443`, redirect `HTTP 80` to HTTPS
- Target group protocol: HTTP
- Target group port: `8000`
- Health check path: `/health`
- Security group inbound: ALB to ECS on `8000`

## 7. Environment Checklist

Required:

- `MONGODB_URI`
- `JWT_SECRET_KEY`
- `CORS_ORIGINS`
- `MODEL_PATH=ml/models/food_model_extensive.pt`
- `CLASS_NAMES_PATH=ml/classes.json`
- `ENVIRONMENT=production`
- `RETRAIN_ON_FEEDBACK=false`

Strongly recommended:

- `CLOUDINARY_CLOUD_NAME`
- `CLOUDINARY_API_KEY`
- `CLOUDINARY_API_SECRET`

Cloudinary is important because ECS task local files are not permanent storage.

## 8. Verify Deployment

```bash
curl https://api.your-domain.com/health
curl https://api.your-domain.com/api/v1/predictions/supported-foods
```

Then set the frontend API URL:

```bash
NEXT_PUBLIC_API_URL=https://api.your-domain.com/api/v1
```

## 9. Updating The Service

For each backend release:

```bash
export IMAGE_TAG=$(git rev-parse --short HEAD)
./scripts/build_push_ecr.sh
```

Update the task definition image tag, register a new revision, then update the service:

```bash
aws ecs update-service \
  --cluster foodintel-cluster \
  --service foodintel-api \
  --task-definition foodintel-api \
  --force-new-deployment \
  --region "$AWS_REGION"
```

## Production Notes

- Keep training datasets out of the Docker image.
- Keep only `ml/models/food_model_extensive.pt` and `ml/classes.json` in the image.
- Use CloudWatch logs for debugging startup/model-load issues.
- Use MongoDB Atlas network access rules that allow ECS egress, or use a NAT/static egress setup if you need strict IP allowlisting.
- Do not enable automatic retraining inside ECS until you have persistent storage and a controlled retraining job.
