"""
api/routes.py
-------------
FastAPI router for POST /v1/quote — the single agent-facing endpoint.

Request lifecycle:
  1. l402_gate dependency fires first.  If the agent hasn't paid, they get a
     402 challenge and this handler never runs.
  2. The QuoteRequest body is validated by Pydantic.
  3. Three async tasks fire simultaneously via asyncio.gather:
       A. fetch_gas_snapshots()    — gas prices for all requested chains
       B. oneinch.fetch_quotes()   — single-chain DEX routes per chain
       C. lifi.fetch_quotes()      — single- and cross-chain bridge routes
  4. All three are wrapped with return_exceptions=True — a total aggregator
     failure becomes a logged warning, not a 500.
  5. Quotes from both aggregators are flattened into one list and fed to
     score_routes() with the gas snapshot attached.
  6. select_optimal_route() picks the winner.
  7. OptimalRouteResponse is returned as JSON.

Design invariants:
  - Zero LLM on this path: every operation is deterministic math or HTTP.
  - If 1inch is down but Li.Fi succeeds, the agent still gets a valid answer.
  - If ALL aggregators fail, the response has optimal_route=null with a clear
    meta.warning — the agent can retry or fall back.
  - Request latency is measured and returned in meta.latency_ms so agents
    can adapt their retry strategy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.aggregator import gas as gas_module
from app.aggregator import lifi as lifi_module
from app.aggregator import oneinch as oneinch_module
from app.optimizer.models import (
    GasSnapshot,
    OptimalRouteResponse,
    QuoteRequest,
    RouteBreakdownResponse,
    RouteQuote,
)
from app.optimizer.tec import score_routes, select_optimal_route, slippage_audit
from app.payment.l402 import l402_gate

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared httpx client — reused across requests for connection pooling.
# Created at module level; lifespan hooks in main.py open/close it.
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the module-level shared client, or a new one if not initialised."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(follow_redirects=True, timeout=10.0)
    return _http_client


async def close_http_client() -> None:
    """Called from main.py lifespan shutdown to drain connections cleanly."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Aggregator fan-out helpers
# ---------------------------------------------------------------------------

async def _safe_gas(
    req: QuoteRequest,
    client: httpx.AsyncClient,
) -> dict:
    """
    Fetch gas snapshots and catch all exceptions.

    Returns an empty dict on failure — missing gas snapshots reduce
    auditability but do not block scoring (gas_usd is already embedded
    in each RouteQuote by the aggregator that fetched it).
    """
    try:
        return await gas_module.fetch_gas_snapshots(req.chains, client=client)
    except Exception as exc:
        logger.warning("Gas fetch failed (non-fatal): %s", exc)
        return {}


async def _safe_oneinch(
    req: QuoteRequest,
    client: httpx.AsyncClient,
) -> list[RouteQuote]:
    """Fetch 1inch quotes, returning an empty list on any failure."""
    try:
        return await oneinch_module.fetch_quotes(req, client=client)
    except Exception as exc:
        logger.warning("1inch fetch failed (non-fatal): %s", exc)
        return []


async def _safe_lifi(
    req: QuoteRequest,
    client: httpx.AsyncClient,
) -> list[RouteQuote]:
    """Fetch Li.Fi quotes, returning an empty list on any failure."""
    try:
        return await lifi_module.fetch_quotes(req, client=client)
    except Exception as exc:
        logger.warning("Li.Fi fetch failed (non-fatal): %s", exc)
        return []


def _serialise_gas_snapshot(snapshots: dict) -> list[dict]:
    """Convert {Chain: GasSnapshot} to a JSON-serialisable list for the response."""
    result = []
    for chain, snap in snapshots.items():
        result.append({
            "chain":                    chain.value,
            "base_fee_gwei":            snap.base_fee_gwei,
            "priority_fee_gwei":        snap.priority_fee_gwei,
            "eth_price_usd":            snap.eth_price_usd,
            "gas_usd_per_unit":         snap.gas_usd_per_unit,
            "l1_surcharge_multiplier":  snap.l1_surcharge_multiplier,
        })
    return result


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post(
    "/v1/quote",
    response_model=OptimalRouteResponse,
    summary="Get optimal cross-chain execution route",
    description=(
        "Returns the mathematically optimal execution route for a token swap "
        "across one or more EVM chains.  Requires L402 Lightning payment. "
        "All computation is deterministic — zero LLM, zero heuristics."
    ),
    responses={
        200: {"description": "Optimal route found (or null if no viable routes exist)"},
        402: {"description": "Payment required — see WWW-Authenticate header for Lightning invoice"},
        422: {"description": "Invalid request body"},
        503: {"description": "All aggregators failed — retry shortly"},
    },
)
async def get_optimal_quote(
    req: QuoteRequest,
    _gate: None = Depends(l402_gate),
) -> OptimalRouteResponse:
    """
    Core handler.  The l402_gate dependency already ran (and passed) by the
    time this function body executes.
    """
    start_ms = int(time.time() * 1000)
    client   = get_http_client()

    logger.info(
        "Quote request | %s→%s %.4f | chains=%s",
        req.token_in, req.token_out, req.amount_in,
        [c.value for c in req.chains],
    )

    # -----------------------------------------------------------------------
    # Step 1 — fan-out all data fetches in parallel
    # -----------------------------------------------------------------------
    # asyncio.gather returns results in the same order as the coroutines.
    # We use the _safe_* wrappers so that any single source failure is
    # isolated and logged rather than propagated as an exception here.
    gas_snapshots, oneinch_quotes, lifi_quotes = await asyncio.gather(
        _safe_gas(req, client),
        _safe_oneinch(req, client),
        _safe_lifi(req, client),
    )

    # -----------------------------------------------------------------------
    # Step 2 — merge quote lists
    # -----------------------------------------------------------------------
    all_quotes: list[RouteQuote] = oneinch_quotes + lifi_quotes

    aggregator_counts = {
        "1inch": len(oneinch_quotes),
        "lifi":  len(lifi_quotes),
        "total": len(all_quotes),
    }

    logger.info(
        "Quotes fetched | 1inch=%d lifi=%d total=%d",
        len(oneinch_quotes), len(lifi_quotes), len(all_quotes),
    )

    # -----------------------------------------------------------------------
    # Step 3 — guard: no quotes at all
    # -----------------------------------------------------------------------
    if not all_quotes:
        logger.error(
            "All aggregators returned zero quotes for %s→%s on chains=%s",
            req.token_in, req.token_out, [c.value for c in req.chains],
        )
        elapsed_ms = int(time.time() * 1000) - start_ms
        return OptimalRouteResponse(
            optimal_route=None,
            all_routes=[],
            gas_snapshot=_serialise_gas_snapshot(gas_snapshots),
            meta={
                "token_in":   req.token_in,
                "token_out":  req.token_out,
                "amount_in":  req.amount_in,
                "chains":     [c.value for c in req.chains],
                "latency_ms": elapsed_ms,
                "aggregator_counts": aggregator_counts,
                "warning": (
                    "All aggregators returned zero quotes. "
                    "The token pair may be unsupported on the requested chains, "
                    "or all aggregators are temporarily unavailable."
                ),
            },
        )

    # -----------------------------------------------------------------------
    # Step 4 — score every quote against the TEC formula
    # -----------------------------------------------------------------------
    now_ms = int(time.time() * 1000)
    scored = score_routes(all_quotes, req, now_ms=now_ms)

    viable_count    = sum(1 for s in scored if s.is_viable)
    discarded_count = len(scored) - viable_count

    logger.info(
        "Scoring complete | total=%d viable=%d discarded=%d",
        len(scored), viable_count, discarded_count,
    )

    # -----------------------------------------------------------------------
    # Step 5 — run slippage audit (non-blocking, for monitoring)
    # -----------------------------------------------------------------------
    # This cross-checks aggregator-reported slippage against our pool-depth
    # model.  Divergences are logged but never block the response.
    try:
        audit = slippage_audit(all_quotes, req)
        diverged = [r for r in audit if r["diverged"]]
        if diverged:
            logger.warning(
                "Slippage divergence detected on %d/%d routes: %s",
                len(diverged), len(audit),
                [(r["route_id"], r["quoted_bps"], r["estimated_bps"]) for r in diverged],
            )
    except Exception as exc:
        logger.debug("Slippage audit error (non-fatal): %s", exc)

    # -----------------------------------------------------------------------
    # Step 6 — select optimal and build response
    # -----------------------------------------------------------------------
    optimal = select_optimal_route(scored)
    elapsed_ms = int(time.time() * 1000) - start_ms

    all_route_responses = [RouteBreakdownResponse.from_scored(s) for s in scored]
    optimal_response    = RouteBreakdownResponse.from_scored(optimal) if optimal else None

    if optimal:
        logger.info(
            "Optimal route | source=%s %s→%s score=%.4f profit_usd=%.4f latency=%dms",
            optimal.quote.source.value,
            optimal.quote.chain_in.value,
            optimal.quote.chain_out.value,
            optimal.profit_score,
            optimal.profit_score,
            elapsed_ms,
        )
    else:
        logger.warning(
            "No viable routes for %s→%s | discarded=%d | latency=%dms",
            req.token_in, req.token_out, discarded_count, elapsed_ms,
        )

    meta: dict[str, Any] = {
        "token_in":          req.token_in,
        "token_out":         req.token_out,
        "amount_in":         req.amount_in,
        "amount_in_usd":     req.amount_in_usd,
        "chains":            [c.value for c in req.chains],
        "latency_ms":        elapsed_ms,
        "aggregator_counts": aggregator_counts,
        "viable_routes":     viable_count,
        "discarded_routes":  discarded_count,
    }

    if not optimal:
        meta["warning"] = (
            f"No viable routes found. {discarded_count} route(s) were discarded. "
            "Try relaxing max_slippage_bps or max_bridge_time_seconds."
        )

    return OptimalRouteResponse(
        optimal_route=optimal_response,
        all_routes=all_route_responses,
        gas_snapshot=_serialise_gas_snapshot(gas_snapshots),
        meta=meta,
    )
