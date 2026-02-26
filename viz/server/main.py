"""FastAPI application for muninn visualization."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import databases, graph, health, kg, vss
from server.services.db import close_connection

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — close DB on shutdown."""
    log.info("Starting muninn-viz server")
    yield
    log.info("Shutting down muninn-viz server")
    close_connection()


app = FastAPI(
    title="muninn-viz",
    description="Interactive visualization for the muninn SQLite extension",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(databases.router)
app.include_router(vss.router)
app.include_router(graph.router)
app.include_router(kg.router)
