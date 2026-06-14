"""
main.py
-------
FastAPI application entry point.

Responsibilities:
  - Instantiate the FastAPI app with LLM-optimised OpenAPI metadata.
  - Register the lifespan context (open/close the shared httpx client).
  - Mount the /v1 router and the machine-discovery (LLMO) router.
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
    PUBLIC_URL              — Canonical public URL of this deployment (for discovery endpoints)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.discovery import discovery_router
from app.api.routes import close_http_client, router

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars may be set directly

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
    _warn_missing_env()
    logger.info("Optimizer API starting up")
    yield
    await close_http_client()
    logger.info("Optimizer API shut down cleanly")


def _warn_missing_env() -> None:
    required = {
        "COINOS_API_KEY": "Lightning invoice generation will fail on /v1/quote",
        "L402_SECRET":    "L402 payment gate will fail to mint/verify macaroons",
    }
    optional = {
        "ONEINCH_API_KEY": "1inch quotes may be rate-limited without an API key",
        "PUBLIC_URL":      "Discovery endpoints will use placeholder base URL",
    }
    for var, consequence in required.items():
        if not os.environ.get(var):
            logger.warning("Missing required env var %s — %s", var, consequence)
    for var, consequence in optional.items():
        if not os.environ.get(var):
            logger.info("Optional env var %s not set — %s", var, consequence)


# ---------------------------------------------------------------------------
# App instantiation — rich OpenAPI metadata for LLM discovery
# ---------------------------------------------------------------------------
# Every string here is written to be consumed by an LLM reading the OpenAPI
# spec, not just a human reading the Swagger UI.  Precision over brevity.

_PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://optimizer-api.example.com")

app = FastAPI(
    title="Cross-Chain Liquidity & Execution Route Optimizer",

    # summary appears in LLM tool-selection contexts (short, directive)
    summary=(
        "Find the cheapest cross-chain EVM token swap route. "
        "Zero LLM, deterministic math, L402 Lightning payment."
    ),

    # description is the full context an LLM reads to decide when/how to use this API.
    # Uses Markdown so Swagger UI renders it nicely; LLMs also parse MD well.
    description="""\
## What this API does

Given a token swap intent, this API queries **1inch** and **Li.Fi** aggregators in
parallel, then scores every route using the **True Execution Cost (TEC)** formula:

```
TEC   = C_slippage + C_gas + C_bridge_fee + C_time_penalty
SCORE = (amount_out × target_price_usd) − TEC − amount_in_usd
```

The route with the highest `SCORE` is returned as `optimal_route`.

**Zero LLM at runtime** — every number is derived from live on-chain data.

---

## Who should call this API

Autonomous AI agents and trading bots that need to:
- Execute a DeFi token swap on any EVM chain
- Compare same-chain vs cross-chain costs mathematically
- Get slippage estimates before signing a transaction

---

## Payment: L402 Lightning Protocol

This API charges **10 satoshis** (≈ $0.006) per query via the L402 standard.

### Agent payment flow

```
1. POST /v1/quote  →  HTTP 402
   WWW-Authenticate: L402 macaroon=<token>, invoice=<BOLT11>

2. Pay the BOLT11 Lightning invoice.
   Your wallet returns a 32-byte preimage (proof of payment).

3. POST /v1/quote  (same body)
   Authorization: L402 <macaroon>:<preimage_hex>
   →  HTTP 200  (route data)
```

Macaroon TTL: **600 seconds** from payment.

---

## Supported chains & tokens

**Chains:** `ethereum`, `arbitrum`, `base`, `optimism`, `polygon`

**Tokens:** `ETH`, `USDC`, `USDT`, `DAI`, `WETH`

---

## Machine discovery

| Resource | URL |
|----------|-----|
| LLM guide (llmstxt.org) | `/llms.txt` |
| AI plugin manifest | `/.well-known/ai-plugin.json` |
| Agent manifest | `/.well-known/agent-manifest.json` |
| Schema.org structured data | `/schema.json` |
""",

    version="0.1.0",

    # contact and license surface in openapi.json — read by tool registries
    contact={
        "name":  "Optimizer API Support",
        "email": "peretzbatel123@gmail.com",
    },

    # Terms surface in ChatGPT plugin review and some agent directories
    terms_of_service=f"{_PUBLIC_URL}/info",

    # servers block is critical: tells LLM clients the actual base URL to call
    servers=[
        {
            "url": _PUBLIC_URL,
            "description": "Production",
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development",
        },
    ],

    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,

    # OpenAPI 3.1 tags with descriptions — shown in Swagger and read by LLMs
    openapi_tags=[
        {
            "name": "Optimizer",
            "description": (
                "Core route optimization endpoint. Requires L402 Lightning payment. "
                "Returns True Execution Cost analysis across all aggregators."
            ),
        },
        {
            "name": "System",
            "description": "Health checks and API metadata. No payment required.",
        },
        {
            "name": "Discovery",
            "description": (
                "Machine-discovery endpoints for AI agents, LLMs, and crawlers. "
                "No payment required. Includes llms.txt, ai-plugin.json, robots.txt."
            ),
        },
    ],
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
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
    No payment required. Used by Railway/Fly.io health checks and agent pre-flight.
    """
    return {"status": "ok"}


@app.get("/info", tags=["System"])
async def info() -> dict:
    """
    Returns API metadata and current pricing.

    No payment required. Agents should call this before /v1/quote to discover:
    - Current price in satoshis (may change)
    - Macaroon TTL (how long after payment the token remains valid)
    - Supported chains and tokens
    """
    from app.payment.coinos import PRICE_SATS
    from app.payment.l402 import MACAROON_TTL_SECONDS

    return {
        "name":                 "Cross-Chain Route Optimizer",
        "version":              "0.1.0",
        "price_sats_per_query": PRICE_SATS,
        "price_usd_approx":     round(PRICE_SATS * 0.0006, 4),  # rough at BTC=$60k
        "macaroon_ttl_seconds": MACAROON_TTL_SECONDS,
        "payment_protocol":     "L402 (Lightning Network)",
        "supported_chains":     ["ethereum", "arbitrum", "base", "optimism", "polygon"],
        "supported_tokens":     ["ETH", "USDC", "USDT", "DAI", "WETH"],
        "aggregators":          ["1inch", "lifi"],
        "discovery": {
            "llms_txt":       "/llms.txt",
            "ai_plugin":      "/.well-known/ai-plugin.json",
            "agent_manifest": "/.well-known/agent-manifest.json",
            "openapi":        "/openapi.json",
            "schema_org":     "/schema.json",
        },
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(router, tags=["Optimizer"])
app.include_router(discovery_router)
