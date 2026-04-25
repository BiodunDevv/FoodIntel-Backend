# FoodIntel Architecture

Text diagram:

`Next.js frontend -> FastAPI backend -> MongoDB`

`FastAPI backend -> PyTorch inference service -> trained checkpoint`

`FastAPI backend -> local uploads or Cloudinary`

Backend responsibilities:

- authentication and JWT issuance
- user profile management
- foods catalog and nutrition lookup
- image upload, inference, and meal persistence
- weekly report aggregation
- Swagger/OpenAPI documentation and standalone HTML status page

ML responsibilities:

- dataset preparation
- transfer-learning training
- evaluation artifacts and local prediction testing

Notes:

- MongoDB is accessed asynchronously through Motor.
- The backend starts even when the trained model is missing.
- The prediction route returns `503` until a model is trained and loaded.
