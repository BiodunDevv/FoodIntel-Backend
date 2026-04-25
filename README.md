# FoodIntel Backend

This folder is ready to be used as a separate deployment repo for the FastAPI API and ML inference service.

## What To Include In The Backend Repo

Keep these items in the backend deployment repo:

- `app/`
- `requirements.txt`
- `render.yaml`
- `.gitignore`
- `tests/` (optional but useful)
- `ml/models/food_model_extensive.pt`
- `ml/classes.json`

The API needs the model checkpoint and class names for inference. It does not need the full training dataset in production.

## Render Deployment

This repo includes `render.yaml` for a Render web service.

Important production note:

- Keep `RETRAIN_ON_FEEDBACK=false` on the public API service
- Do feedback collection in production, but retrain offline or in a separate background worker

## Environment Variables

Set these in Render:

- `MONGODB_URI`
- `JWT_SECRET_KEY`
- `CORS_ORIGINS`

Optional:

- `MONGODB_DB_NAME`
- `MAX_UPLOAD_SIZE_MB`

Defaults in `render.yaml` already set:

- `ENVIRONMENT=production`
- `UPLOAD_DIR=uploads`
- `MODEL_PATH=ml/models/food_model_extensive.pt`
- `CLASS_NAMES_PATH=ml/classes.json`
- `RETRAIN_ON_FEEDBACK=false`

## Local Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Push As A Separate Repo

From inside this folder after you copy in the `ml/` inference files:

```bash
git init
git add .
git commit -m "Initial FoodIntel backend"
git branch -M main
git remote add origin <YOUR_BACKEND_REPO_URL>
git push -u origin main
```

## Suggested Hosting Layout

- Frontend: Vercel
- Backend API: Render Web Service
- Database: MongoDB Atlas
- Retraining: separate worker or offline job, not the public web service
# FoodIntel-Backend
