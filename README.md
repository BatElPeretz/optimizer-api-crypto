# Cross-Chain Liquidity & Execution Route Optimizer

> **B2A API** — built for autonomous AI agents and trading bots.  
> **Zero LLM at runtime** — deterministic math on live on-chain data.  
> **L402 Lightning paywall** — 10 sats (~$0.006) per query.

[![License: MIT + Commons Clause](https://img.shields.io/badge/License-MIT%20%2B%20Commons%20Clause-red.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![L402](https://img.shields.io/badge/auth-L402%20Lightning-orange.svg)](https://docs.lightning.engineering/the-lightning-network/l402)

---

## What it does

Given a token swap intent (`ETH → USDC`, `5 ETH`, chains: `arbitrum + base`), this API:

1. Queries **1inch v5.2** (single-chain DEX aggregator) and **Li.Fi** (cross-chain bridge + DEX) in parallel
2. Scores every route using the **True Execution Cost (TEC)** formula:

```
TEC   = C_slippage + C_gas + C_bridge_fee + C_time_penalty
SCORE = (amount_out × target_price_usd) − TEC − amount_in_usd
```

3. Returns the highest-scoring viable route + full cost breakdown of all candidates

**Supported chains:** Ethereum · Arbitrum · Base · Optimism · Polygon  
**Supported tokens:** ETH · USDC · USDT · DAI · WETH

---

## Quick start (agent integration)

### Step 1 — Discover pricing

```bash
curl https://optimizer-api-crypto.onrender.com/info
```

```json
{
  "price_sats_per_query": 10,
  "price_usd_approx": 0.006,
  "macaroon_ttl_seconds": 600,
  "supported_chains": ["ethereum", "arbitrum", "base", "optimism", "polygon"]
}
```

### Step 2 — Get a Lightning invoice (402 challenge)

```bash
curl -X POST https://optimizer-api-crypto.onrender.com/v1/quote \
  -H "Content-Type: application/json" \
  -d '{"token_in":"ETH","token_out":"USDC","amount_in":5.0,"amount_in_usd":18500,"target_price_usd":1.0,"chains":["arbitrum","base"]}'
```

```
HTTP/1.1 402 Payment Required
WWW-Authenticate: L402 macaroon=eyJ..., invoice=lnbc100n1p...
```

### Step 3 — Pay the invoice, then retry

```bash
# After paying: your Lightning wallet returns a preimage (32-byte hex)
curl -X POST https://optimizer-api-crypto.onrender.com/v1/quote \
  -H "Content-Type: application/json" \
  -H "Authorization: L402 eyJ...:<preimage_hex>" \
  -d '{"token_in":"ETH","token_out":"USDC","amount_in":5.0,"amount_in_usd":18500,"target_price_usd":1.0,"chains":["arbitrum","base"]}'
```

### Step 4 — Parse the response

```json
{
  "optimal_route": {
    "source": "lifi",
    "chain_in": "arbitrum",
    "chain_out": "base",
    "amount_out": 18380.5,
    "true_execution_cost_usd": 0.82,
    "slippage_usd": 0.50,
    "gas_usd": 0.02,
    "bridge_fee_usd": 0.30,
    "profit_score": -120.3,
    "is_viable": true
  },
  "all_routes": [ ... ],
  "meta": { "latency_ms": 340, "viable_routes": 4 }
}
```

---

## Machine discovery

| Resource | URL |
|----------|-----|
| LLM guide (`llmstxt.org`) | `/llms.txt` |
| OpenAI-compatible plugin | `/.well-known/ai-plugin.json` |
| Agent manifest (L402-native) | `/.well-known/agent-manifest.json` |
| Schema.org structured data | `/schema.json` |
| Full OpenAPI 3.1 spec | `/openapi.json` |
| Swagger UI | `/docs` |

---

## Architecture

```
POST /v1/quote
       │
       ├─ l402_gate()          ← verify Lightning payment
       │
       └─ asyncio.gather()
              ├─ gas.py         ← EIP-1559 fee history RPC per chain
              ├─ oneinch.py     ← 1inch v5.2 single-chain quotes
              └─ lifi.py        ← Li.Fi cross-chain quotes (N² pairs)
                     │
              score_routes()    ← TEC formula, 5 viability checks
                     │
              select_optimal()  ← argmax(profit_score)
```

## Running locally

```bash
git clone https://github.com/BatElPeretz/optimizer-api-crypto
cd cross-chain-optimizer
pip install -r requirements.txt
cp .env.example .env   # fill in COINOS_API_KEY and L402_SECRET
uvicorn app.main:app --reload --port 8000
```

```bash
pytest tests/ -v   # 127 tests
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `COINOS_API_KEY` | Yes | Coinos Lightning API token (`coinos.io → Settings → API`) |
| `L402_SECRET` | Yes | ≥32-char secret for macaroon signing |
| `ONEINCH_API_KEY` | No | 1inch API key (rate-limited without it) |
| `PUBLIC_URL` | No | Canonical public URL (used in discovery endpoints) |
| `OPTIMIZER_PRICE_SATS` | No | Satoshis per query (default: 10) |
| `MACAROON_TTL_SECONDS` | No | Macaroon validity in seconds (default: 600) |

---

## Keywords

`defi` `cross-chain` `route-optimization` `ethereum` `arbitrum` `base` `optimism` `polygon`
`1inch` `lifi` `token-swap` `slippage` `gas-optimization` `true-execution-cost` `tec`
`l402` `lightning-network` `bitcoin` `micropayments` `b2a-api` `agentic-api`
`autonomous-agent` `ai-agent` `langchain` `autogpt` `crewai` `llmo`
`fastapi` `python` `pydantic` `asyncio` `httpx`
