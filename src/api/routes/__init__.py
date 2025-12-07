"""
API route modules.

Provides FastAPI routers for different API endpoints.
"""

from src.api.routes.documents import router as documents_router
from src.api.routes.tasks import router as tasks_router
from src.api.routes.health import router as health_router
from src.api.routes.schemas import router as schemas_router


__all__ = [
    "documents_router",
    "tasks_router",
    "health_router",
    "schemas_router",
]
