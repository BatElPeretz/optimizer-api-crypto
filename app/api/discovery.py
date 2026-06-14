"""
api/discovery.py
----------------
Machine-discovery layer (Agentic SEO / LLMO).

All endpoints here are unauthenticated and require no payment.
Their sole purpose is to make the API perfectly legible to:
  - AI agent orchestrators (LangChain, AutoGPT, CrewAI, etc.)
  - LLM tool registries (OpenAI plugins, Anthropic tool-use)
  - Web crawlers that index AI APIs (GPTBot, Claude-Web, Perplexity-Bot)
  - Developer discovery (DevDocs, RapidAPI, public OpenAPI indexes)

Endpoints:
  GET /.well-known/ai-plugin.json      — OpenAI-compatible plugin manifest
  GET /.well-known/agent-manifest.json — Emerging open agent manifest standard
  GET /llms.txt                        — llmstxt.org markdown guide for LLMs
  GET /robots.txt                      — Crawler permissions (allow AI bots)
  GET /sitemap.xml                     — API sitemap for indexing
  GET /schema.json                     — Schema.org SoftwareApplication structured data
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse, Response

discovery_router = APIRouter(tags=["Discovery"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    """Return the canonical public URL of the API (set via PUBLIC_URL env var)."""
    return os.environ.get("PUBLIC_URL", "https://optimizer-api.example.com").rstrip("/")


# ---------------------------------------------------------------------------
# /.well-known/ai-plugin.json
# OpenAI-compatible plugin manifest — read by ChatGPT plugins, many agent
# frameworks, and LLM tool-discovery crawlers.
# ---------------------------------------------------------------------------

@discovery_router.get(
    "/.well-known/ai-plugin.json",
    include_in_schema=False,  # meta-endpoint, skip from main OpenAPI spec
)
async def ai_plugin_manifest() -> JSONResponse:
    """
    OpenAI-compatible plugin manifest.

    This file is the standard entry point that agent orchestrators look for
    when discovering API capabilities.  It explains what the API does, how
    to authenticate, and where to find the full OpenAPI spec.

    L402 authentication is declared here so agents built with payment-aware
    frameworks (e.g. L402-capable LangChain forks) know up-front that they
    need Lightning Network payment capability before calling /v1/quote.
    """
    base = _base_url()
    return JSONResponse({
        "schema_version": "v1",
        "name_for_human": "Cross-Chain Route Optimizer",
        "name_for_model": "cross_chain_route_optimizer",

        # Short description shown in model's tool selector UI.
        "description_for_human": (
            "Find the mathematically optimal cross-chain token swap route across "
            "Ethereum, Arbitrum, Base, Optimism, and Polygon.  Pays for itself "
            "in milliseconds via Lightning Network (L402)."
        ),

        # Longer description the LLM reads to decide WHEN to call this tool.
        # Written in directive style — tells the model exactly what situations
        # should trigger a call to this API.
        "description_for_model": (
            "Use this tool when an autonomous agent needs to execute a DeFi token "
            "swap and wants the lowest True Execution Cost (TEC) route across multiple "
            "EVM chains.  "
            "The tool compares routes from 1inch and Li.Fi aggregators, then applies "
            "the TEC formula: TEC = slippage_cost + gas_cost + bridge_fee + time_penalty. "
            "It returns a ranked list of routes with full cost breakdowns so the agent "
            "can make an informed execution decision.\n\n"
            "WHEN TO CALL:\n"
            "- Agent is about to execute a token swap on any EVM chain\n"
            "- Agent needs to know if cross-chain routing is cheaper than same-chain\n"
            "- Agent wants to compare 1inch vs Li.Fi routes mathematically\n"
            "- Agent needs slippage estimates before committing to a transaction\n\n"
            "PAYMENT (L402 Lightning):\n"
            "1. Call POST /v1/quote WITHOUT an Authorization header.\n"
            "2. Server returns HTTP 402 with header: "
            "WWW-Authenticate: L402 macaroon=<token>, invoice=<BOLT11>\n"
            "3. Pay the BOLT11 Lightning invoice (typically 10 sats ≈ $0.006).\n"
            "4. Retry POST /v1/quote with header: "
            "Authorization: L402 <macaroon>:<preimage_hex>\n"
            "5. The server verifies SHA256(preimage) == payment_hash embedded in macaroon.\n\n"
            "DO NOT CALL if the agent has no Lightning Network payment capability."
        ),

        "auth": {
            "type": "oauth",
            # We declare oauth as the closest standard type, but clarify in
            # description_for_model that the actual mechanism is L402/Lightning.
            # True L402 support is described in the agent-manifest.json.
            "authorization_url": f"{base}/docs",
            "scope": "l402",
            "verification_tokens": {},
        },

        "api": {
            "type": "openapi",
            "url": f"{base}/openapi.json",
            "is_user_authenticated": False,
        },

        "logo_url": f"{base}/logo.png",
        "contact_email": "peretzbatel123@gmail.com",
        "legal_info_url": f"{base}/info",
    })


# ---------------------------------------------------------------------------
# /.well-known/agent-manifest.json
# Richer, L402-native manifest for next-generation agentic frameworks.
# Includes: tool schema, payment protocol, capability declarations,
# supported chains, tokens, and aggregators.
# ---------------------------------------------------------------------------

@discovery_router.get(
    "/.well-known/agent-manifest.json",
    include_in_schema=False,
)
async def agent_manifest() -> JSONResponse:
    base = _base_url()
    return JSONResponse({
        "manifest_version": "1.0",
        "id": "cross-chain-route-optimizer",
        "name": "Cross-Chain Liquidity & Execution Route Optimizer",
        "description": (
            "Deterministic, zero-LLM cross-chain swap route optimizer. "
            "Computes True Execution Cost across 1inch and Li.Fi aggregators "
            "and returns the mathematically optimal route."
        ),
        "version": "0.1.0",
        "base_url": base,
        "docs_url": f"{base}/docs",
        "openapi_url": f"{base}/openapi.json",
        "llms_txt_url": f"{base}/llms.txt",

        # Payment protocol declaration — frameworks that understand L402 can
        # automatically pay and retry without human intervention.
        "payment": {
            "protocol": "L402",
            "network": "lightning",
            "currency": "BTC",
            "price_sats": 10,
            "price_usd_approx": 0.006,
            "flow": [
                "POST /v1/quote  →  HTTP 402 + WWW-Authenticate: L402 macaroon=<token>, invoice=<BOLT11>",
                "Pay BOLT11 invoice on Lightning Network",
                "POST /v1/quote  →  Authorization: L402 <macaroon>:<preimage_hex>  →  HTTP 200",
            ],
        },

        "capabilities": {
            "zero_llm": True,
            "deterministic": True,
            "cross_chain": True,
            "real_time_gas": True,
            "slippage_validation": True,
        },

        "supported_chains": ["ethereum", "arbitrum", "base", "optimism", "polygon"],
        "supported_tokens": ["ETH", "USDC", "USDT", "DAI", "WETH"],
        "aggregators": ["1inch", "lifi"],

        "tools": [
            {
                "name": "get_optimal_swap_route",
                "endpoint": "POST /v1/quote",
                "payment_required": True,
                "description": (
                    "Returns the optimal cross-chain swap route ranked by "
                    "True Execution Cost (TEC = slippage + gas + bridge_fee + time_penalty)."
                ),
                "input_schema": f"{base}/openapi.json#/components/schemas/QuoteRequest",
                "output_schema": f"{base}/openapi.json#/components/schemas/OptimalRouteResponse",
            }
        ],

        "contact": {
            "email": "peretzbatel123@gmail.com",
        },
    })


# ---------------------------------------------------------------------------
# /llms.txt
# Markdown guide specifically written for LLMs.
# Spec: https://llmstxt.org — plain text discovered by AI crawlers at /llms.txt
# The format mirrors README but is optimised for token efficiency and model
# comprehension rather than human aesthetics.
# ---------------------------------------------------------------------------

_LLMS_TXT = """\
# Cross-Chain Liquidity & Execution Route Optimizer

> **B2A API** — built for autonomous AI agents, not humans.
> **Zero LLM at runtime** — all computation is deterministic math on on-chain data.
> **Monetised via L402** — Lightning Network micropayments (≈$0.006 per query).

## What this API does

Given a token swap intent (`ETH → USDC`, `5 ETH`, `ethereum + arbitrum + base`),
this API:

1. Fetches live quotes from **1inch** (single-chain DEX aggregator) and **Li.Fi** (cross-chain bridge + DEX).
2. Scores every route using the **True Execution Cost (TEC)** formula:
   ```
   TEC   = C_slippage + C_gas + C_bridge + C_time_penalty
   score = (amount_out × target_price_usd) - TEC - amount_in_usd
   ```
3. Returns the **highest-scoring viable route** plus all candidates ranked.

## When to call this API

Call this API when:
- You are about to execute a DeFi token swap on any EVM chain.
- You want to know whether routing cross-chain is cheaper than same-chain.
- You need a slippage estimate before signing a transaction.
- You need to compare 1inch vs Li.Fi routes with an objective cost metric.

Do NOT call this API if you have no Lightning Network payment capability.

## Authentication: L402 Lightning Protocol

This API uses the **L402 standard** (IETF draft: HTTP 402 + Lightning).

### Step-by-step payment flow

```
1. POST /v1/quote
   Body: { "token_in": "ETH", "token_out": "USDC", ... }
   → HTTP 402
     WWW-Authenticate: L402 macaroon=<base64url_token>, invoice=<BOLT11_string>

2. Pay the BOLT11 invoice on the Lightning Network.
   Amount: typically 10 satoshis ≈ $0.006 USD.
   Your Lightning wallet returns a 32-byte preimage (proof of payment).

3. POST /v1/quote  (same body, add Authorization header)
   Authorization: L402 <macaroon>:<preimage_hex>
   → HTTP 200  (route data)
```

The macaroon expires in **10 minutes** (600 seconds).
If you receive HTTP 401 with `{"detail": "macaroon expired"}`, start over from step 1.

### Parsing the WWW-Authenticate header

```
WWW-Authenticate: L402 macaroon=eyJ..., invoice=lnbc100n1p...

Split on ", invoice=" to separate the two values.
macaroon  = everything after "macaroon=" up to ", invoice="
invoice   = the BOLT11 string starting with "lnbc"
preimage  = 32-byte hex string returned by your Lightning wallet after payment
```

### Authorization header format

```
Authorization: L402 <macaroon_base64url>:<preimage_hex_64chars>
```

## Endpoint: POST /v1/quote

**Requires payment.** Returns the optimal swap route.

### Request body (JSON)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `token_in` | string | yes | Input token symbol. Supported: `ETH`, `USDC`, `USDT`, `DAI`, `WETH` |
| `token_out` | string | yes | Output token symbol. Must differ from `token_in` |
| `amount_in` | float | yes | Input amount in token units (e.g. `5.0` for 5 ETH). Must be > 0 |
| `amount_in_usd` | float | yes | USD value of `amount_in` at call time. Agent pre-computes this from a price oracle |
| `target_price_usd` | float | yes | Spot price of `token_out` in USD (e.g. `1.0` for USDC) |
| `chains` | array[string] | yes | Chains to search. Any subset of: `ethereum`, `arbitrum`, `base`, `optimism`, `polygon` |
| `max_slippage_bps` | int | no | Max acceptable slippage in basis points. Default: `100` (1%). Range: 1–2000 |
| `agent_time_value_per_second` | float | no | USD cost of 1 second of waiting (bridge time penalty coefficient). Default: `0.0001` |
| `max_bridge_time_seconds` | int | no | Hard limit on cross-chain bridge time. Default: `900` (15 min). Routes above this are excluded |

### Minimal example request

```json
{
  "token_in": "ETH",
  "token_out": "USDC",
  "amount_in": 5.0,
  "amount_in_usd": 18500.0,
  "target_price_usd": 1.0,
  "chains": ["arbitrum", "base"]
}
```

### Response structure

```json
{
  "optimal_route": {
    "route_id": "lifi_arbitrum_base_ETH_USDC_1718000000",
    "source": "lifi",
    "route_type": "bridge_then_swap",
    "chain_in": "arbitrum",
    "chain_out": "base",
    "token_in": "ETH",
    "token_out": "USDC",
    "amount_out": 18380.5,
    "amount_out_min": 18196.7,
    "effective_rate": 3676.1,
    "profit_score": -120.3,
    "net_output_value_usd": 18380.5,
    "true_execution_cost_usd": 0.82,
    "slippage_usd": 0.50,
    "gas_usd": 0.02,
    "bridge_fee_usd": 0.30,
    "time_penalty_usd": 0.00,
    "slippage_bps": 50,
    "pool_liquidity_usd": 12000000.0,
    "bridge_time_seconds": 120,
    "is_viable": true,
    "disqualification_reason": null
  },
  "all_routes": [ ... ],
  "gas_snapshot": [ ... ],
  "meta": {
    "token_in": "ETH",
    "token_out": "USDC",
    "amount_in": 5.0,
    "chains": ["arbitrum", "base"],
    "latency_ms": 340,
    "viable_routes": 4,
    "discarded_routes": 1
  }
}
```

`optimal_route` is `null` when no route passes viability checks — relax
`max_slippage_bps` or `max_bridge_time_seconds` and retry.

## Free endpoints (no payment)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe. Returns `{"status": "ok"}` |
| `GET /info` | Pricing, supported chains, tokens, aggregators |
| `GET /openapi.json` | Full OpenAPI 3.1 specification |
| `GET /docs` | Swagger UI |

## Error reference

| HTTP status | Meaning | Action |
|-------------|---------|--------|
| 402 | Payment required | Read `WWW-Authenticate`, pay invoice, retry with `Authorization` header |
| 401 | Bad credentials | Macaroon expired or preimage wrong — restart L402 flow from step 1 |
| 422 | Invalid request body | Fix the field listed in `detail` |
| 503 | All aggregators down | Retry after 5–10 seconds |

## Cost model

- **Price**: 10 satoshis per query (≈ $0.006 at BTC = $60,000)
- **Macaroon TTL**: 600 seconds (10 minutes from payment)
- **Latency**: typically 200–500 ms (parallel aggregator fan-out)
- **Freshness**: quotes are discarded after 3 seconds — you always get live data

## Support

- Docs: /docs
- OpenAPI spec: /openapi.json
- Contact: peretzbatel123@gmail.com
"""


@discovery_router.get(
    "/llms.txt",
    include_in_schema=False,
    response_class=PlainTextResponse,
)
async def llms_txt() -> PlainTextResponse:
    """
    Machine-readable API guide following the llmstxt.org specification.
    AI crawlers (Claude-Web, GPTBot, Perplexity-Bot) index this file to
    teach the LLM how and when to call this API.
    """
    return PlainTextResponse(_LLMS_TXT, media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# /robots.txt
# Explicitly permit AI crawlers so they index our llms.txt and openapi.json.
# Most AI training crawlers respect robots.txt.
# ---------------------------------------------------------------------------

_ROBOTS_TXT = """\
User-agent: *
Allow: /

# Explicitly allow all known AI agent and LLM crawlers
User-agent: GPTBot
Allow: /

User-agent: ChatGPT-User
Allow: /

User-agent: Claude-Web
Allow: /

User-agent: anthropic-ai
Allow: /

User-agent: PerplexityBot
Allow: /

User-agent: Applebot
Allow: /

User-agent: cohere-ai
Allow: /

User-agent: meta-externalagent
Allow: /

# Key discovery files — always crawlable
Allow: /llms.txt
Allow: /.well-known/ai-plugin.json
Allow: /.well-known/agent-manifest.json
Allow: /openapi.json
Allow: /sitemap.xml
Allow: /schema.json

Sitemap: {base_url}/sitemap.xml
"""


@discovery_router.get(
    "/robots.txt",
    include_in_schema=False,
    response_class=PlainTextResponse,
)
async def robots_txt() -> PlainTextResponse:
    content = _ROBOTS_TXT.replace("{base_url}", _base_url())
    return PlainTextResponse(content, media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# /sitemap.xml
# Helps search engines and API indexers discover all endpoints.
# ---------------------------------------------------------------------------

@discovery_router.get(
    "/sitemap.xml",
    include_in_schema=False,
)
async def sitemap() -> Response:
    base = _base_url()
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:api="https://www.api-sitemap.org/schemas/sitemap/0.9">
  <url>
    <loc>{base}/llms.txt</loc>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base}/.well-known/ai-plugin.json</loc>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base}/.well-known/agent-manifest.json</loc>
    <changefreq>monthly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>{base}/openapi.json</loc>
    <changefreq>weekly</changefreq>
    <priority>0.9</priority>
  </url>
  <url>
    <loc>{base}/docs</loc>
    <changefreq>weekly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>{base}/schema.json</loc>
    <changefreq>monthly</changefreq>
    <priority>0.7</priority>
  </url>
  <url>
    <loc>{base}/info</loc>
    <changefreq>weekly</changefreq>
    <priority>0.6</priority>
  </url>
</urlset>"""
    return Response(content=xml, media_type="application/xml")


# ---------------------------------------------------------------------------
# /schema.json
# Schema.org SoftwareApplication structured data.
# Helps Google, Bing, and emerging "AI API indexes" understand what this is.
# Also consumed by tools like Toolhouse, Composio, and Wordware's tool registry.
# ---------------------------------------------------------------------------

@discovery_router.get(
    "/schema.json",
    include_in_schema=False,
)
async def schema_org() -> JSONResponse:
    base = _base_url()
    return JSONResponse({
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        "name": "Cross-Chain Liquidity & Execution Route Optimizer",
        "applicationCategory": "FinanceApplication",
        "applicationSubCategory": "DeFi API",
        "description": (
            "B2A (Business-to-Agent) API for autonomous AI agents. "
            "Finds the mathematically optimal cross-chain EVM token swap route "
            "using True Execution Cost (TEC) analysis across 1inch and Li.Fi aggregators. "
            "Zero LLM at runtime — all computation is deterministic on-chain math. "
            "Monetised via L402 Lightning Network micropayments."
        ),
        "url": base,
        "softwareVersion": "0.1.0",
        "operatingSystem": "Any",
        "offers": {
            "@type": "Offer",
            "price": "0.006",
            "priceCurrency": "USD",
            "description": "10 satoshis per API query (≈ $0.006 at BTC=$60k)",
        },
        "featureList": [
            "Cross-chain route optimization (Ethereum, Arbitrum, Base, Optimism, Polygon)",
            "True Execution Cost (TEC) formula: slippage + gas + bridge fee + time penalty",
            "Real-time gas price normalization via EIP-1559 fee history RPC",
            "1inch v5.2 aggregator integration",
            "Li.Fi cross-chain bridge aggregator integration",
            "L402 Lightning Network payment gate",
            "Slippage validation against pool depth model",
        ],
        "audience": {
            "@type": "Audience",
            "audienceType": "Autonomous AI Agents, DeFi developers, Trading bots",
        },
        "author": {
            "@type": "Person",
            "email": "peretzbatel123@gmail.com",
        },
        "documentation": f"{base}/docs",
        "isAccessibleForFree": False,
        "paymentAccepted": "Bitcoin Lightning Network (L402)",
        "keywords": [
            "DeFi", "cross-chain", "route optimization", "1inch", "Li.Fi",
            "Ethereum", "Arbitrum", "Base", "L402", "Lightning Network",
            "AI agent", "autonomous agent", "token swap", "slippage",
            "gas optimization", "TEC", "execution cost", "B2A API",
            "LLMO", "agentic API",
        ],
    })
