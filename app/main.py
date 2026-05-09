from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config.settings import BACKEND_DIR, get_settings
from app.database.mongodb import mongo_manager
from app.routes.auth_routes import router as auth_router
from app.routes.food_routes import router as food_router
from app.routes.meal_routes import router as meal_router
from app.routes.admin_routes import router as admin_router
from app.routes.page_routes import router as page_router
from app.routes.prediction_routes import router as prediction_router
from app.routes.report_routes import router as report_router
from app.routes.user_routes import router as user_router
from app.services.ml_service import ml_service
from app.services.scheduler_service import daily_scheduler


settings = get_settings()
settings.upload_path.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    await mongo_manager.connect(settings)
    ml_service.load()
    daily_scheduler.start()
    yield
    daily_scheduler.stop()
    await mongo_manager.disconnect()


app = FastAPI(
    title="FoodIntel API",
    description=(
        "FoodIntel is an intelligent food recognition and nutritional analysis backend "
        "built with FastAPI, MongoDB, and PyTorch. It supports user authentication, "
        "image-based food prediction, nutrition estimation, health scoring, recommendations, "
        "meal history, and weekly reports. Nutritional outputs are estimates and must not be treated as medical advice."
    ),
    version="1.0.0",
    contact={"name": "FoodIntel Project Team", "email": "project@example.com"},
    license_info={"name": "MIT"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Health", "description": "Operational health and environment status."},
        {"name": "Authentication", "description": "User registration, login, and current user endpoints."},
        {"name": "Users", "description": "Authenticated user profile management."},
        {"name": "Foods", "description": "Food catalog and development seed data."},
        {"name": "Predictions", "description": "Image upload and ML-powered food predictions."},
        {"name": "Meals", "description": "Meal history operations."},
        {"name": "Reports", "description": "Weekly nutrition reporting."},
        {"name": "Project Page", "description": "Standalone HTML project overview page."},
    ],
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BACKEND_DIR / "app/static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(settings.upload_path), check_dir=False), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def error_response(message: str, detail: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "message": message, "detail": detail},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict):
        message = str(exc.detail.get("message", "Request failed."))
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "message": message, "detail": exc.detail},
        )
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
    message = detail if exc.status_code < 500 else "Server error."
    return error_response(message, detail, exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return error_response("Validation error.", str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY)


@app.exception_handler(RuntimeError)
async def runtime_exception_handler(_: Request, exc: RuntimeError) -> JSONResponse:
    return error_response("Service unavailable.", str(exc), status.HTTP_503_SERVICE_UNAVAILABLE)


app.include_router(page_router)
app.include_router(admin_router)
app.include_router(auth_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")
app.include_router(food_router, prefix="/api/v1")
app.include_router(prediction_router, prefix="/api/v1")
app.include_router(meal_router, prefix="/api/v1")
app.include_router(report_router, prefix="/api/v1")
