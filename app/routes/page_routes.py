from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config.settings import BACKEND_DIR, get_settings
from app.database.mongodb import mongo_manager
from app.services.ml_service import ml_service
from app.services.retraining_service import retraining_service


templates = Jinja2Templates(directory=str(BACKEND_DIR / "app/templates"))
router = APIRouter(tags=["Project Page"])


@router.get(
    "/",
    response_class=HTMLResponse,
    summary="Serve the FoodIntel project page",
    description="Render the project status page with live API, database, and model state.",
)
async def project_page(request: Request) -> HTMLResponse:
    settings = get_settings()
    context = {
        "request": request,
        "environment": settings.environment,
        "database_connected": mongo_manager.connected,
        "database_error": mongo_manager.last_error,
        "model_status": ml_service.status(),
        "retraining_status": retraining_service.status(),
    }
    return templates.TemplateResponse("index.html", context)


@router.get(
    "/health",
    summary="Get health status",
    description="Return the API, database, model, and environment status.",
    tags=["Health"],
)
async def health() -> dict:
    settings = get_settings()
    return {
        "success": True,
        "message": "Health status retrieved successfully.",
        "data": {
            "api": "ok",
            "database_connected": mongo_manager.connected,
            "database_error": mongo_manager.last_error,
            "model_loaded": ml_service.model_loaded,
            "model_version": ml_service.model_version,
            "retraining": retraining_service.status(),
            "environment": settings.environment,
        },
    }
