# FoodIntel Testing

## Manual checklist

1. Create `.env` from `.env.example` and add `MONGODB_URI`.
2. Install backend dependencies.
3. Run `uvicorn backend.app.main:app --reload`.
4. Open `http://localhost:8000` and verify the project page loads.
5. Open `http://localhost:8000/health` and confirm the backend responds.
6. Open `http://localhost:8000/docs`.
7. Register a new user with `POST /api/v1/auth/register`.
8. Login with `POST /api/v1/auth/login`.
9. Authorize Swagger with `Bearer <access_token>`.
10. Call `POST /api/v1/foods/seed` in development mode.
11. Call `GET /api/v1/foods`.
12. Train a model with `python ml/train.py`.
13. Run `python ml/evaluate.py`.
14. Upload a supported image to `POST /api/v1/predictions/image`.
15. Confirm `GET /api/v1/meals` returns the saved meal.
16. Confirm `GET /api/v1/reports/weekly` returns aggregate data.

## Automated tests

The included pytest coverage focuses on:

- auth service helpers
- health score and recommendation logic
- nutrition calculation
- weekly report generation
- application health endpoint behavior
