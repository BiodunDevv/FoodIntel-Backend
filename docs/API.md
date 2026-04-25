# FoodIntel API

Base prefix: `/api/v1`

Core groups:

- Authentication: register, login, current user
- Users: profile update
- Foods: list, detail, development seed
- Predictions: upload image for inference
- Meals: history, detail, delete
- Reports: weekly nutrition summary
- Health and project page: `/health`, `/`, `/docs`, `/redoc`, `/openapi.json`

Swagger flow:

1. Start the backend.
2. Open `http://localhost:8000/docs`.
3. Register a user through `POST /api/v1/auth/register`.
4. Login through `POST /api/v1/auth/login`.
5. Copy the `access_token`.
6. Click `Authorize`.
7. Paste `Bearer <access_token>`.
8. Seed foods with `POST /api/v1/foods/seed`.
9. Test foods, predictions, meals, and reports.

All success responses follow:

```json
{
  "success": true,
  "message": "Human-readable message",
  "data": {}
}
```

All error responses follow:

```json
{
  "success": false,
  "message": "Human-readable error message",
  "detail": "Optional technical detail"
}
```
