"""
main.py
-------
FastAPI application entry point.

Responsibilities:
  - Instantiate the FastAPI app with metadata agents will see in /openapi.json.
  - Register the lifespan context (open/close the shared httpx client).
  - Mount the /v1 router.
  - Add CORS middleware (agents may be cross-origin HTTP clients).
  - Provide /health and /info endpoints (no payment required).

Running locally:
    uvicorn app.main:app --reload --port 8000

Environment variables (set in .env):
    COINOS_API_KEY          — Coinos Lightning API token (required for /v1/quote)
    L402_SECRET             — ≥32-char secret for macaroon signing (required)
    ONEINCH_API_KEY         — 1inch API token (optional; rate-limited without it)
    OPTIMIZER_PRICE_SATS    — Satoshis per quote request (default: 10)
    MACAROON_TTL_SECONDS    — Macaroon validity window (default: 600)
    LOG_LEVEL               — Python logging level (default: INFO)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import close_http_client, router

# Load .env before anything reads os.environ
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — shared resource management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages resources that should live for the full server lifetime.

    On startup:
      - Log confirmation that env vars are present (not their values).

    On shutdown:
      - Drain the shared httpx connection pool cleanly so no in-flight
        requests are abandoned during a graceful restart.
    """
    # --- startup ---
    _warn_missing_env()
    logger.info("Optimizer API starting up")
    yield
    # --- shutdown ---
    await close_http_client()
    logger.info("Optimizer API shut down cleanly")


def _warn_missing_env() -> None:
    """Log warnings for required environment variables that are missing."""
    required = {
        "COINOS_API_KEY": "Lightning invoice generation will fail on /v1/quote",
        "L402_SECRET":    "L402 payment gate will fail to mint/verify macaroons",
    }
    optional = {
        "ONEINCH_API_KEY": "1inch quotes may be rate-limited without an API key",
    }
    for var, consequence in required.items():
        if not os.environ.get(var):
            logger.warning("Missing required env var %s — %s", var, consequence)
    for var, consequence in optional.items():
        if not os.environ.get(var):
            logger.info("Optional env var %s not set — %s", var, consequence)


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cross-Chain Liquidity & Execution Route Optimizer",
    description=(
        "B2A (Business-to-Agent) API for finding the mathematically optimal "
        "cross-chain token swap route across Ethereum, Arbitrum, and Base. "
        "All computation is deterministic — zero LLM, zero heuristics. "
        "Monetised via L402 Lightning Network micropayments (Coinos)."
    ),
    version="0.1.0",
    docs_url="/docs",       # Swagger UI — useful during development
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    # Agents may call from any origin.  Tighten this for production if needed.
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
    expose_headers=["WWW-Authenticate"],  # agents must be able to read the 402 header
)


# ---------------------------------------------------------------------------
# Unprotected utility endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> dict:
    """
    Liveness probe — returns 200 as long as the process is running.
    No payment required.  Used by Railway/Fly.io health checks.
    """
    return {"status": "ok"}


@app.get("/info", tags=["System"])
async def info() -> dict:
    """
    Returns API metadata and current pricing.
    No payment required.  Agents can call this to discover the cost before
    committing to a /v1/quote request.
    """
    from app.payment.coinos import PRICE_SATS
    from app.payment.l402 import MACAROON_TTL_SECONDS

    return {
        "name":                 "Cross-Chain Route Optimizer",
        "version":              "0.1.0",
        "price_sats_per_query": PRICE_SATS,
        "macaroon_ttl_seconds": MACAROON_TTL_SECONDS,
        "supported_chains":     ["ethereum", "arbitrum", "base", "optimism", "polygon"],
        "supported_tokens":     ["ETH", "USDC", "USDT", "DAI", "WETH"],
        "aggregators":          ["1inch", "lifi"],
        "payment_protocol":     "L402 (Lightning Network)",
        "docs":                 "/docs",
    }


# ---------------------------------------------------------------------------
# Protected router
# ---------------------------------------------------------------------------

app.include_router(router, tags=["Optimizer"])
