# FoodIntel — Backend

> AI-powered food recognition and nutrition analysis API — built with FastAPI, Motor (async MongoDB), PyTorch, and Pydantic v2.

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Environment Variables](#environment-variables)
5. [Getting Started](#getting-started)
6. [API Reference](#api-reference)
7. [ML Pipeline](#ml-pipeline)
8. [Supported Food Classes](#supported-food-classes)
9. [Prediction Confidence System](#prediction-confidence-system)
10. [Health Score Engine](#health-score-engine)
11. [Retraining Pipeline](#retraining-pipeline)
12. [Database Schema](#database-schema)
13. [Services](#services)
14. [Deployment](#deployment)
15. [Ethical Notice](#ethical-notice)

---

## Overview

The FoodIntel backend is a production-ready REST API that:

1. Authenticates users with JWT (register, login, profile management)
2. Accepts food images (file upload or remote URL), runs PyTorch inference, and returns structured nutrition data
3. Persists every prediction as a `meal_log` document in MongoDB with nutrition, health score, and recommendations
4. Exposes meal history with date-range filtering and monthly reporting
5. Collects user prediction feedback (correct / wrong food) and queues it for model retraining review
6. Serves a live project status page at `/` with real-time API, database, and model state

The backend is designed to **fail gracefully**: if the ML model is not loaded, prediction routes return a clean `503`. If an image is unclear or outside the model's vocabulary, a structured `422` is returned with the best-guess label, confidence, and threshold details — never a silent wrong answer.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| Async MongoDB driver | Motor + PyMongo |
| Data validation | Pydantic v2 |
| Authentication | Python-JOSE (JWT HS256) + Passlib (bcrypt) |
| ML inference | PyTorch + torchvision (MobileNetV3 / EfficientNet-B0) |
| Image processing | Pillow |
| HTTP client | httpx (async image-URL fetching) |
| Config | pydantic-settings (`Settings` class from `.env`) |
| Template rendering | Jinja2 (project status page) |
| ASGI server | Uvicorn |

---

## Project Structure

```
foodintel-backend/
├── app/
│   ├── main.py                     # FastAPI app, middleware, exception handlers
│   ├── config/
│   │   └── settings.py             # Pydantic Settings — reads .env
│   ├── database/
│   │   └── mongodb.py              # MongoManager — connect/disconnect lifecycle
│   ├── models/
│   │   └── meal_model.py           # build_meal_document() factory
│   ├── routes/
│   │   ├── auth_routes.py          # /auth/register, /auth/login, /auth/me
│   │   ├── user_routes.py          # PATCH /users/me
│   │   ├── food_routes.py          # GET /foods, seed endpoint
│   │   ├── meal_routes.py          # GET/DELETE /meals, GET /meals/:id
│   │   ├── prediction_routes.py    # POST /predictions/image, /image-url, /feedback
│   │   ├── report_routes.py        # GET /reports/weekly (date-range)
│   │   └── page_routes.py          # GET / (status page), GET /health
│   ├── schemas/
│   │   ├── common.py               # APIResponse[T], ErrorResponse, PyObjectId
│   │   ├── auth_schema.py
│   │   ├── user_schema.py
│   │   ├── food_schema.py          # NutritionValues
│   │   ├── meal_schema.py          # MealLogPublic, MealListPayload
│   │   ├── prediction_schema.py    # PredictionResult, PredictionFeedbackRequest
│   │   └── report_schema.py        # WeeklyReportPayload, DailyCalorieTrend
│   ├── services/
│   │   ├── auth_service.py         # JWT creation/verification, get_current_user dep
│   │   ├── ml_service.py           # MLService singleton — load, predict, status
│   │   ├── health_score_service.py # Rule-based scoring (0–100)
│   │   ├── nutrition_service.py    # Per-serving nutrition calculation
│   │   ├── recommendation_service.py # Text recommendations from nutrition data
│   │   ├── report_service.py       # build_report_for_range()
│   │   ├── retraining_service.py   # Full background retrain pipeline + hot-reload
│   │   ├── scheduler_service.py    # Daily midnight UTC auto-retrain scheduler
│   │   ├── storage_service.py      # Local file save / Cloudinary fallback
│   │   └── gamification_service.py # XP event definitions
│   ├── utils/
│   │   ├── image.py                # open_image, open_image_from_url, validate_upload
│   │   └── object_id.py            # serialize_mongo_document, to_object_id
│   ├── static/
│   │   └── styles.css              # Project status page styles
│   └── templates/
│       └── index.html              # Jinja2 status page template
│
├── ml/
│   ├── train.py                    # Full training pipeline (ImageFolder)
│   ├── evaluate.py                 # Per-class accuracy report
│   ├── predict_local.py            # CLI single-image inference
│   ├── export_feedback_dataset.py  # Export approved feedback → training data
│   ├── build_master_dataset.py     # Merge Nigerian + Food-101 datasets
│   ├── build_clean_dataset.py      # Deduplicate and validate image folders
│   ├── prepare_imagefolder.py      # Organise raw images into ImageFolder layout
│   ├── prepare_nigerian_dataset.py # Process Nigerian food image sources
│   ├── prepare_merged_nigerian_dataset.py
│   ├── download_food101.py         # torchvision Food-101 downloader
│   ├── run_local_pipeline.py       # End-to-end: prepare → train → evaluate
│   ├── common.py                   # Shared paths, transform definitions
│   └── models/
│       ├── food_model_extensive.pt # Trained model checkpoint (not in git)
│       └── classes.json            # Ordered list of class slugs
│
├── tests/
│   ├── conftest.py
│   ├── test_health_route.py
│   ├── test_auth_utils.py
│   └── test_core_services.py
│
├── requirements.txt
├── render.yaml                     # Render deployment config
└── .env.example
```

---

## Environment Variables

Create `foodintel-backend/.env` from `.env.example`:

```env
# Application
APP_NAME=FoodIntel
ENVIRONMENT=development          # development | production
SECRET_KEY=your-jwt-secret-key   # min 32 chars — use secrets.token_hex(32)
ACCESS_TOKEN_EXPIRE_MINUTES=10080  # 7 days

# MongoDB
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=foodintel

# CORS — JSON array of allowed origins
CORS_ORIGINS=["http://localhost:3000"]

# File uploads
UPLOAD_PATH=./uploads
MAX_UPLOAD_SIZE_MB=10

# ML model
MODEL_PATH=./ml/models/food_model_extensive.pt
MODEL_CLASSES_PATH=./ml/models/classes.json
MODEL_ARCHITECTURE=mobilenet_v3_large   # mobilenet_v3_small | mobilenet_v3_large | efficientnet_b0 | resnet18
CONFIDENCE_THRESHOLD=0.50
MARGIN_THRESHOLD=0.15

# Retraining
RETRAIN_ON_FEEDBACK=true
RETRAIN_MIN_FEEDBACK_SAMPLES=5      # minimum approved samples before auto-retrain fires
RETRAIN_EPOCHS=2
RETRAIN_DEVICE=cpu                  # cpu | cuda | auto

# Admin panel (used by /admin/* endpoints and the Next.js /admin-access page)
ADMIN_SECRET=foodintel-admin-2024   # change this in production!

# Cloudinary (optional — used if all three are set)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- MongoDB running locally (`mongod`) or a MongoDB Atlas connection string
- (Optional) A trained model checkpoint at `ml/models/food_model_extensive.pt`

### Install and run

```bash
cd foodintel-backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in your values
uvicorn app.main:app --reload --port 8000
```

Open:

- **Swagger UI** — [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc** — [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Status page** — [http://localhost:8000](http://localhost:8000)
- **Health JSON** — [http://localhost:8000/health](http://localhost:8000/health)

### Seed food data (development)

```bash
# With the backend running, call the seed endpoint:
curl -X POST http://localhost:8000/api/v1/foods/seed \
  -H "Authorization: Bearer <admin_token>"
```

This populates the `foods` collection with nutrition data for all 27 supported food classes.

### Run tests

```bash
pytest tests/ -v
```

---

## API Reference

All endpoints are prefixed with `/api/v1`. Every response follows the envelope:

```json
{
  "success": true,
  "message": "Human-readable status message.",
  "data": { ... }
}
```

Errors follow:

```json
{
  "success": false,
  "message": "Human-readable error.",
  "detail": "string or structured object"
}
```

### Authentication — `/auth`

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | No | Create account — `{ full_name, email, password }` |
| POST | `/auth/login` | No | Login — returns `{ token, user }` |
| GET | `/auth/me` | Bearer | Get current authenticated user |

### Users — `/users`

| Method | Path | Auth | Description |
|---|---|---|---|
| PATCH | `/users/me` | Bearer | Update profile — `{ full_name?, age?, height_cm?, weight_kg?, goal?, activity_level? }` |

### Foods — `/foods`

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/foods` | No | List all food records in the catalogue |
| GET | `/foods/{slug}` | No | Get single food by slug (e.g. `jollof_rice_nigeria`) |
| POST | `/foods/seed` | Bearer | Seed the foods collection (development only) |

### Predictions — `/predictions`

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/predictions/image` | Bearer | Upload image file (`multipart/form-data`) — `file` + optional `serving_size_g` |
| POST | `/predictions/image-url` | Bearer | Predict from remote URL — `{ image_url, serving_size_g? }` |
| POST | `/predictions/feedback` | Bearer | Submit prediction correction — `{ meal_id, is_correct, corrected_slug?, notes? }` |
| GET | `/predictions/supported-foods` | No | List model classes with slugs and display labels |

**Low-confidence 422 response** (when `confidence < CONFIDENCE_THRESHOLD`):

```json
{
  "success": false,
  "detail": {
    "code": "unsupported_or_uncertain_image",
    "message": "I am not confident this image is one of the supported food classes.",
    "top_prediction": "rice",
    "confidence": 0.31,
    "runner_up": "beans",
    "runner_up_confidence": 0.18,
    "margin": 0.13,
    "confidence_threshold": 0.50,
    "margin_threshold": 0.15
  }
}
```

### Meals — `/meals`

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/meals` | Bearer | List meals — `?limit=500&skip=0&date=YYYY-MM-DD` |
| GET | `/meals/{id}` | Bearer | Get single meal log |
| DELETE | `/meals/{id}` | Bearer | Delete a meal log |

`limit` maximum is `500`. When `date` is provided, results are sorted ascending (chronological for that day) and the effective limit is raised to 500 regardless of the `limit` parameter.

### Reports — `/reports`

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/reports/weekly` | Bearer | Nutrition report — `?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD` (defaults to current calendar month) |

**Report response includes:**

- `total_meals`, `avg_calories_per_day`, `avg_health_score`
- `most_frequent_food` — top logged food slug + label
- `daily_calorie_trend` — one entry per day in the range with `date`, `calories`, `meal_count`
- `macro_totals` — summed protein, carbs, fat, fibre across the period
- `recommendations` — text insights from the scoring engine
- `start_date`, `end_date` — the resolved range

---

## ML Pipeline

### Model architecture

The model is a standard transfer-learning classifier built on top of one of four backbones:

| Architecture | Parameter count | Notes |
|---|---|---|
| `mobilenet_v3_small` | ~2.5 M | Fastest — suitable for CPU inference |
| `mobilenet_v3_large` | ~5.5 M | Default production choice — good speed/accuracy balance |
| `efficientnet_b0` | ~5.3 M | Slightly higher accuracy ceiling |
| `resnet18` | ~11 M | Baseline comparator |

The final classification layer is replaced with a `Linear(in_features, num_classes)` layer matching the number of food classes. The checkpoint is saved with `torch.save({ "model_state_dict": ..., "model_name": ..., "classes": [...], "version": ... })`.

### Training

```bash
cd foodintel-backend
python ml/train.py \
  --data-dir ./ml/data/merged_dataset \
  --model-name mobilenet_v3_large \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --output-dir ./ml/models
```

The training script:
1. Loads an `ImageFolder` dataset from `--data-dir`
2. Applies standard augmentation: random horizontal flip, colour jitter, random crop, normalisation
3. Fine-tunes with Adam + cosine LR schedule
4. Saves the best checkpoint (by validation accuracy) as `food_model_extensive.pt`
5. Writes `classes.json` alongside the checkpoint

### Dataset preparation

```bash
# Download Food-101 baseline
python ml/download_food101.py

# Prepare Nigerian food images
python ml/prepare_nigerian_dataset.py

# Merge Nigerian + Food-101 subsets into a unified ImageFolder
python ml/build_master_dataset.py

# Clean duplicates and validate file integrity
python ml/build_clean_dataset.py
```

### Local inference test

```bash
python ml/predict_local.py --image path/to/food.jpg
```

### Feedback export for retraining

```bash
python ml/export_feedback_dataset.py \
  --output-dir ./ml/data/feedback_export \
  --status approved
```

This pulls all `approved` or `auto_approved` feedback documents from MongoDB and copies the source images into an `ImageFolder`-compatible directory structure organised by corrected slug.

---

## Supported Food Classes

The current model recognises **27 foods**, focused on Nigerian and West African cuisine with selected international staples.

| # | Display Name | Slug |
|---|---|---|
| 1 | Abacha | `abacha` |
| 2 | Afang Soup | `afang_soup` |
| 3 | Akara | `akara` |
| 4 | Amala | `amala` |
| 5 | Asaro | `asaro` |
| 6 | Banga Soup | `banga_soup` |
| 7 | Beans | `beans` |
| 8 | Boli | `boli` |
| 9 | Chin Chin | `chin_chin` |
| 10 | Edikaikong Soup | `edikaikong_soup` |
| 11 | Egusi Soup | `egusi_soup` |
| 12 | Ewedu Soup | `ewedu_soup` |
| 13 | Jollof Rice Nigeria | `jollof_rice_nigeria` |
| 14 | Masa | `masa` |
| 15 | Meat Pie | `meat_pie` |
| 16 | Moi Moi | `moi_moi` |
| 17 | Nkwobi | `nkwobi` |
| 18 | Ogbono Soup | `ogbono_soup` |
| 19 | Oha Soup | `oha_soup` |
| 20 | Okro Soup | `okro_soup` |
| 21 | Pepper Soup | `pepper_soup` |
| 22 | Plantain | `plantain` |
| 23 | Puff Puff | `puff_puff` |
| 24 | Rice and Stew | `rice_and_stew` |
| 25 | Suya | `suya` |
| 26 | Vegetable Soup | `vegetable_soup` |
| 27 | Yam | `yam` |

> **Growth:** More Nigerian and African meals will be added as reviewed user-submitted images are collected and folded back into retraining cycles. The feedback system described below is the primary collection mechanism.

---

## Prediction Confidence System

The model does not guess blindly. Every prediction goes through a two-gate confidence check before a meal is saved:

### Gate 1 — Absolute confidence

`confidence >= CONFIDENCE_THRESHOLD` (default `0.50`)

The top softmax probability must be at least 50%. A model that has never seen a particular food will typically assign low, spread-out probabilities — this gate catches those cases.

### Gate 2 — Margin

`(top_confidence - runner_up_confidence) >= MARGIN_THRESHOLD` (default `0.15`)

Even if the top confidence is above 50%, the model must be clearly more confident about the top class than the second-best class. This prevents borderline cases where the model is genuinely confused between two similar foods.

### When both gates pass

The meal is persisted, nutrition is looked up from the `foods` collection, and a `MealLogPublic` is returned.

### When either gate fails

A `422 Unprocessable Entity` is returned with the structured `detail` object. The image is **not** saved as a meal. The frontend displays the "Food not identified" banner with the best-guess information so the user understands why the scan failed and what to try instead.

This system is the primary reason FoodIntel is honest about uncertainty rather than confidently wrong.

---

## Health Score Engine

`app/services/health_score_service.py` computes a score from 0–100 for each meal using rule-based deductions:

| Condition | Deduction | Reason text |
|---|---|---|
| Calories > 700 kcal | −20 | High calorie serving |
| Calories > 500 kcal | −10 | Moderately high calories |
| Protein < 10 g | −15 | Low protein for satiety |
| Fibre < 5 g | −15 | Low fibre, less filling |
| Sodium > 800 mg | −20 | High sodium |
| Sodium > 500 mg | −10 | Moderately high sodium |
| Fat > 30 g | −10 | High fat content |
| Carbs > 60 g AND protein < 15 g | −10 | High-carb, low-protein balance |

The score is clamped to `[0, 100]`. All deduction reason texts are included in the `recommendations` array on the meal response.

---

## Retraining Pipeline

### Feedback collection

Every result page shows a feedback card with two options:

- **"Yes, looks right"** → `is_correct: true`, stored as `status: confirmed`
- **"Report a correction"** → user provides the correct food name → stored as `status: auto_approved`

All feedback is stored in the `prediction_feedback` MongoDB collection alongside the original image URL, predicted slug, confidence, and any notes.

### Admin review panel

Correction feedback (`auto_approved`) goes to the **Admin Panel** for human review before being used in retraining. The panel lives at:

```
http://localhost:3000/admin-access          # development
https://your-domain.com/admin-access        # production
```

Enter the value of `ADMIN_SECRET` (from your `.env`) to unlock the panel. From there you can:

- **Approve** a feedback item → status changes to `approved`, eligible for the next training run
- **Reject** a feedback item → status changes to `rejected`, permanently excluded
- **Trigger retraining now** → immediately kicks off a background training job with all `approved` samples
- **Force retrain** → bypasses the minimum-sample threshold (useful when you have fewer than `RETRAIN_MIN_FEEDBACK_SAMPLES` but want to test)
- **Live status polling** — the panel auto-polls the retraining job status every 8 seconds while a job is running

> The admin panel is a Next.js page that calls the backend admin JSON API (`/admin/feedback`, `/admin/feedback/{id}/approve`, `/admin/feedback/{id}/reject`, `/admin/retrain`, `/admin/retraining-status`). All endpoints require `?secret=ADMIN_SECRET` and return `403` on mismatch. They are excluded from `/docs`.

### Daily auto-scheduler

`scheduler_service.py` starts a background thread on app startup that calculates the seconds until the next midnight UTC and sleeps precisely that long. On each tick it:

1. Counts all `approved` feedback documents in MongoDB
2. Calls `retraining_service.request_retraining(approved_count)`
3. If the count meets `RETRAIN_MIN_FEEDBACK_SAMPLES`, a training thread is spawned automatically — no human intervention needed

The scheduler stops cleanly during app shutdown via the FastAPI lifespan hook.

### Retraining pipeline (what happens under the hood)

```
approved feedback in MongoDB
        │
        ▼
_prepare_working_dataset()
  - copies ml/dataset_master/ → ml/dataset_live/
  - exports approved images from MongoDB (local path / data URI / HTTP URL)
  - copies them into ml/dataset_live/train/<corrected_slug>/
  - marks exported feedback as used_for_training
        │
        ▼
ml/train.py  (subprocess)
  --model mobilenet_v3_small
  --resume-from <current checkpoint>
  --freeze-backbone  (backbone frozen for first epoch)
  --freeze-epochs 1
        │
        ▼
ml_service.load()   ← hot-reload without server restart
```

The model checkpoint is overwritten in-place; the next prediction immediately uses the updated weights.

---

## Database Schema

All documents stored in MongoDB. Below are the key collections:

### `users`

```json
{
  "_id": "ObjectId",
  "full_name": "string",
  "email": "string (unique)",
  "hashed_password": "string",
  "age": "int | null",
  "height_cm": "float | null",
  "weight_kg": "float | null",
  "goal": "lose_weight | maintain | gain_muscle | eat_healthier | null",
  "activity_level": "sedentary | lightly_active | moderately_active | very_active | null",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### `foods`

```json
{
  "_id": "ObjectId",
  "slug": "string (unique)",
  "name": "string",
  "calories_per_100g": "float",
  "protein_per_100g": "float",
  "carbs_per_100g": "float",
  "fat_per_100g": "float",
  "fibre_per_100g": "float",
  "sodium_per_100g": "float",
  "description": "string | null"
}
```

### `meal_logs`

```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "food_id": "string | null",
  "image_url": "string",
  "predicted_food": "string",
  "predicted_slug": "string",
  "confidence": "float",
  "serving_size_g": "float",
  "nutrition": {
    "calories": "float",
    "protein": "float",
    "carbs": "float",
    "fat": "float",
    "fibre": "float",
    "sodium": "float"
  },
  "health_score": "float",
  "recommendations": ["string"],
  "model_version": "string",
  "feedback": {
    "is_correct": "bool",
    "status": "confirmed | auto_approved",
    "corrected_slug": "string",
    "corrected_food": "string",
    "notes": "string | null",
    "submitted_at": "datetime"
  },
  "created_at": "datetime"
}
```

### `prediction_feedback`

```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "meal_id": "string",
  "image_url": "string",
  "predicted_slug": "string",
  "predicted_food": "string",
  "confidence": "float",
  "is_correct": "bool",
  "corrected_slug": "string",
  "notes": "string | null",
  "status": "confirmed | auto_approved | approved | rejected",
  "created_at": "datetime"
}
```

---

## Services

| Service | File | Responsibility |
|---|---|---|
| `auth_service` | `auth_service.py` | JWT creation, verification, `get_current_user` FastAPI dependency |
| `ml_service` | `ml_service.py` | Singleton — loads model checkpoint at startup, runs inference, manages `LowConfidencePredictionError` |
| `health_score_service` | `health_score_service.py` | Rule-based health score (0–100) + reason text list |
| `nutrition_service` | `nutrition_service.py` | Scales per-100g values to actual serving size |
| `recommendation_service` | `recommendation_service.py` | Generates human-readable text recommendations from nutrition data |
| `report_service` | `report_service.py` | `build_report_for_range(meals, start, end)` — aggregates trends and totals |
| `retraining_service` | `retraining_service.py` | Background pipeline: exports approved feedback, runs `ml/train.py`, hot-reloads model |
| `scheduler_service` | `scheduler_service.py` | Daily midnight UTC background thread that auto-triggers retraining on approved feedback |
| `storage_service` | `storage_service.py` | Saves uploaded files locally; falls back to Cloudinary if `CLOUDINARY_URL` is set |
| `gamification_service` | `gamification_service.py` | XP event constants |

---

## Deployment

### AWS ECS/Fargate (recommended for production)

The production deployment path for FoodIntel is a Docker image on ECS Fargate behind an Application Load Balancer.

Deployment assets are included under:

```text
deploy/aws/
├── README.md                         # End-to-end ECS/Fargate guide
└── ecs-task-definition.template.json # Task definition template

scripts/build_push_ecr.sh             # Build and push Docker image to ECR
Dockerfile                            # Production API container
.dockerignore                         # Keeps datasets, venvs, reports, and secrets out of the image
requirements-api.txt                  # Lean runtime dependencies for deployment
```

Quick path:

```bash
cd foodintel-backend
docker build -t foodintel-api .

export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export IMAGE_TAG=$(git rev-parse --short HEAD)

./scripts/build_push_ecr.sh
```

Then follow [deploy/aws/README.md](deploy/aws/README.md) to create the ECR repository, Parameter Store secrets, ECS task definition, Fargate service, ALB, HTTPS listener, and health checks.

### Render (optional simple backend hosting)

A `render.yaml` is included at the root of the backend folder.

```bash
# Push foodintel-backend/ to a separate Git repo, then connect to Render
# Set all environment variables in the Render dashboard
```

Key Render settings:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Environment:** Python 3.11

### Environment checklist

- [ ] `JWT_SECRET_KEY` is a random 32+ character string
- [ ] `MONGODB_URI` points to Atlas (or a managed MongoDB instance)
- [ ] `CORS_ORIGINS` includes the deployed frontend domain
- [ ] `MODEL_PATH=ml/models/food_model_extensive.pt`
- [ ] `CLASS_NAMES_PATH=ml/classes.json`
- [ ] `ENVIRONMENT=production`
- [ ] `RETRAIN_ON_FEEDBACK=false` for container deployments
- [ ] Cloudinary credentials are configured for persistent production uploads

### Model checkpoint in production

The model file (`food_model_extensive.pt`) must be available at `MODEL_PATH` at startup. For ECS, the production Dockerfile copies `ml/models/food_model_extensive.pt` and `ml/classes.json` into the image so deployments are self-contained and fast.

If the model file is missing, the backend starts successfully and all other endpoints work — prediction routes return a clean `503 Service Unavailable` until the checkpoint is available.

### Docker

```bash
cd foodintel-backend
docker build -t foodintel-api .
docker run -p 8000:8000 --env-file .env foodintel-api
```

---

## Ethical Notice

Nutritional values and health scores produced by FoodIntel are **estimates for educational and personal tracking purposes only**. They are computed from average per-100g values for each food class and scaled to the user's declared serving size. They are not medical advice and must not be used to diagnose, treat, or manage any health condition or nutrient deficiency. Always consult a registered dietitian or physician for personalised nutritional guidance.

The AI model is trained on a curated dataset of Nigerian and international food images. Its predictions reflect the distribution of that training data. Foods outside the supported classes will be rejected with a low-confidence response rather than assigned a wrong label. User feedback is actively collected to improve future model versions.
