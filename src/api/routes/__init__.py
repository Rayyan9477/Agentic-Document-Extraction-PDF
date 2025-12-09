"""
API route modules.

Provides FastAPI routers for different API endpoints.
"""

from src.api.routes.auth import router as auth_router
from src.api.routes.documents import router as documents_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.health import router as health_router
from src.api.routes.schemas import router as schemas_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.queue import router as queue_router


__all__ = [
    "auth_router",
    "documents_router",
    "tasks_router",
    "health_router",
    "schemas_router",
    "dashboard_router",
    "queue_router",
]
