"""FastAPI application for muninn visualization."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes import databases, graph, health, kg, vss
from server.services.db import validate_startup
from server.services.kg import warm_embedding_model

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — validate DB + warm model on startup."""
    log.info("Starting muninn-viz server")
    # Open a temporary connection to validate the DB exists and extension loads.
    # This surfaces failures at startup rather than on first request.
    conn = validate_startup()
    log.info("Database and extension validated")
    # Pre-load the sentence-transformers embedding model so the first KG query
    # doesn't take 5-15 seconds downloading/validating model files.
    warm_embedding_model(conn)
    conn.close()
    yield
    log.info("Shutting down muninn-viz server")


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
