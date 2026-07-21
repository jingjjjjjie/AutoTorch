"""FastAPI application assembly for the visualization platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from advanced_visualization.web.routes import (
    artifacts,
    experiments,
    projections,
    review,
    sources,
)


STATIC_DIR = Path(__file__).with_name("static")


def create_app() -> FastAPI:
    application = FastAPI(title="AutoTorch Visualization", version="1.1.0")
    for route_module in (sources, review, projections, experiments, artifacts):
        application.include_router(route_module.router)
    application.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @application.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "advanced_visualization.web.app:app", host="0.0.0.0", port=8000, reload=False
    )


if __name__ == "__main__":
    main()
