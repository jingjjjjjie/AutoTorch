"""FastAPI application assembly for the visualization platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import Scope

from advanced_visualization.web.routes import (
    artifacts,
    experiments,
    projections,
    review,
    sources,
)


STATIC_DIR = Path(__file__).with_name("static")
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


class FreshStaticFiles(StaticFiles):
    """Serve one coherent ES-module graph after application updates."""

    def is_not_modified(
        self,
        response_headers: Headers,
        request_headers: Headers,
    ) -> bool:
        del response_headers, request_headers
        return False

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(NO_CACHE_HEADERS)
        return response


def create_app() -> FastAPI:
    application = FastAPI(title="AutoTorch Visualization", version="1.1.1")
    for route_module in (sources, review, projections, experiments, artifacts):
        application.include_router(route_module.router)
    application.mount("/assets", FreshStaticFiles(directory=STATIC_DIR), name="assets")

    @application.get("/api/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE_HEADERS)

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
