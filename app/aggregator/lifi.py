"""
aggregator/lifi.py
------------------
Async client for the Li.Fi API (https://li.quest/v1/).

Why Li.Fi is the superior aggregator for cross-chain routes:
  - Aggregates 20+ bridges (Stargate, Across, CCTP, Hop, Synapse, ...)
  - Returns toAmountMin directly — no need to derive slippage ourselves.
  - Provides executionDuration (bridge time in seconds) — feeds our
    time_penalty TEC component directly.
  - Returns itemised gasCosts and feeCosts per step — cleaner than 1inch.
  - No API key required on the free tier.

Li.Fi quote endpoint: GET https://li.quest/v1/quote
  Required params: fromChain, toChain, fromToken, toToken, fromAmount
  Optional params: slippage, order (RECOMMENDED | FASTEST | CHEAPEST | SAFEST)

Response structure (simplified):
{
  "estimate": {
    "toAmount":          "18400000000",    ← raw in toToken decimals
    "toAmountMin":       "18032000000",    ← raw, post-slippage minimum
    "gasCosts":          [{"amountUSD": "0.45"}],
    "feeCosts":          [{"amountUSD": "3.21"}],
    "executionDuration": 120              ← seconds
  },
  "action": {
    "fromChainId":  1,
    "toChainId":    42161,
    ...
  },
  "type": "lifi"
}

Slippage derivation:
  slippage_bps = (toAmount - toAmountMin) / toAmount × 10_000
  This is exact — Li.Fi enforces toAmountMin on-chain.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from app.aggregator.constants import (
    CHAIN_FROM_ID,
    CHAIN_ID,
    LIFI_CHAIN_KEY,
    token_address,
    raw_to_float,
)
from app.optimizer.models import (
    AggregatorSource,
    Chain,
    QuoteRequest,
    RouteQuote,
    RouteType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIFI_BASE_URL   = "https://li.quest/v1"
REQUEST_TIMEOUT = 8.0   # Li.Fi is slower than 1inch (aggregates multiple sources)
MAX_RETRIES     = 2
RETRY_DELAY_BASE = 0.5

# Li.Fi order preference — RECOMMENDED balances cost, speed, and safety.
# CHEAPEST minimises fees but may use slower bridges.
LIFI_ORDER = "RECOMMENDED"

# Default slippage tolerance to pass to Li.Fi (0.5%).
# Li.Fi uses this to compute toAmountMin.  Agents can override per request.
DEFAULT_SLIPPAGE = 0.005   # 0.5% as a decimal


# ---------------------------------------------------------------------------
# Route type classifier
# ---------------------------------------------------------------------------

def _classify_route_type(data: dict[str, Any]) -> RouteType:
    """
    Map Li.Fi's step types to our RouteType enum.

    Li.Fi responses contain a "steps" array.  Each step has a "type":
      "swap"     → on-chain DEX swap
      "cross"    → cross-chain bridge transfer
      "protocol" → protocol-specific action

    Classification logic:
      - All steps are swaps → MULTI_HOP
      - Mix of cross + swap → BRIDGE_THEN_SWAP
      - Single swap step   → DIRECT_SWAP
    """
    steps = data.get("includedSteps", []) or data.get("steps", [])
    step_types = {s.get("type", "") for s in steps}

    if "cross" in step_types and "swap" in step_types:
        return RouteType.BRIDGE_THEN_SWAP
    if "cross" in step_types:
        return RouteType.BRIDGE_THEN_SWAP
    if len(steps) > 1:
        return RouteType.MULTI_HOP
    return RouteType.DIRECT_SWAP


def _extract_gas_usd(estimate: dict[str, Any]) -> float:
    """
    Sum all gasCosts[].amountUSD across every step of the route.
    Li.Fi returns per-step gas costs as strings — we accumulate them.
    Returns 0.0 if gasCosts is absent (some same-chain routes omit it).
    """
    total = 0.0
    for cost in estimate.get("gasCosts", []):
        try:
            total += float(cost.get("amountUSD", 0))
        except (ValueError, TypeError):
            pass
    return total


def _extract_bridge_fee_usd(estimate: dict[str, Any]) -> float:
    """
    Sum all feeCosts[].amountUSD across every step of the route.
    These are the bridge protocol fees (Stargate LP fee, Across relayer fee,
    etc.) — distinct from gas costs.
    """
    total = 0.0
    for cost in estimate.get("feeCosts", []):
        try:
            total += float(cost.get("amountUSD", 0))
        except (ValueError, TypeError):
            pass
    return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_route(
    client: httpx.AsyncClient,
    chain_in: Chain,
    chain_out: Chain,
    req: QuoteRequest,
) -> Optional[RouteQuote]:
    """
    Fetch a single Li.Fi quote for one (chain_in, chain_out) pair.

    Same-chain pairs are valid (Li.Fi handles DEX-only routes).
    Cross-chain pairs trigger Li.Fi's bridge aggregation.

    Returns None on any unrecoverable error.
    """
    try:
        src_addr = token_address(req.token_in, chain_in)
        dst_addr = token_address(req.token_out, chain_out)
    except KeyError as exc:
        logger.debug("Li.Fi | Skipping %s→%s: %s", chain_in.value, chain_out.value, exc)
        return None

    from_chain_id = CHAIN_ID[chain_in]
    to_chain_id   = CHAIN_ID[chain_out]

    # Li.Fi accepts raw integer amount (in fromToken decimals)
    from app.aggregator.constants import token_decimals
    decimals_in = token_decimals(req.token_in)
    raw_amount  = str(int(req.amount_in * (10 ** decimals_in)))

    params = {
        "fromChain":   from_chain_id,
        "toChain":     to_chain_id,
        "fromToken":   src_addr,
        "toToken":     dst_addr,
        "fromAmount":  raw_amount,
        "slippage":    DEFAULT_SLIPPAGE,
        "order":       LIFI_ORDER,
        "allowExchanges": "",   # empty string = allow all DEXes
    }

    url = f"{LIFI_BASE_URL}/quote"

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 404:
                logger.debug(
                    "Li.Fi | No route for %s %s→%s on %s→%s",
                    req.token_in, req.token_out,
                    chain_in.value, chain_out.value,
                    resp.text[:200],
                )
                return None

            if resp.status_code == 400:
                logger.debug("Li.Fi | 400 for %s→%s: %s", chain_in.value, chain_out.value, resp.text[:200])
                return None

            resp.raise_for_status()
            return _parse_response(resp.json(), chain_in, chain_out, req)

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 503):
                wait = RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "Li.Fi | %d on %s→%s, retrying in %.1fs",
                    exc.response.status_code, chain_in.value, chain_out.value, wait,
                )
                await asyncio.sleep(wait)
            elif attempt == MAX_RETRIES:
                logger.warning(
                    "Li.Fi | HTTP %d on %s→%s after %d attempts",
                    exc.response.status_code, chain_in.value, chain_out.value, MAX_RETRIES + 1,
                )
                return None
            else:
                await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == MAX_RETRIES:
                logger.warning("Li.Fi | Connection error %s→%s: %s", chain_in.value, chain_out.value, exc)
                return None
            await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))

        except Exception as exc:
            logger.warning("Li.Fi | Unexpected error %s→%s: %s", chain_in.value, chain_out.value, exc)
            return None

    return None


def _parse_response(
    data: dict[str, Any],
    chain_in: Chain,
    chain_out: Chain,
    req: QuoteRequest,
) -> Optional[RouteQuote]:
    """
    Map a raw Li.Fi quote JSON response to our RouteQuote dataclass.

    Key advantage over 1inch: Li.Fi gives us:
      - toAmountMin  → exact slippage enforced on-chain (no derivation needed)
      - gasCosts[]   → itemised per-step gas in USD
      - feeCosts[]   → itemised bridge fees in USD
      - executionDuration → bridge settlement time in seconds

    Slippage formula (exact, from on-chain enforcement):
        slippage_bps = (toAmount - toAmountMin) / toAmount × 10_000
    """
    try:
        estimate = data["estimate"]

        amount_out     = raw_to_float(estimate["toAmount"], req.token_out)
        amount_out_min = raw_to_float(estimate["toAmountMin"], req.token_out)

        # Exact slippage from Li.Fi's own toAmountMin computation
        if amount_out > 0:
            slippage_bps = int(
                ((amount_out - amount_out_min) / amount_out) * 10_000
            )
        else:
            slippage_bps = 0

        gas_usd        = _extract_gas_usd(estimate)
        bridge_fee_usd = _extract_bridge_fee_usd(estimate)
        bridge_time    = int(estimate.get("executionDuration", 0))

        route_type = _classify_route_type(data)

        # Pool liquidity: Li.Fi doesn't expose it directly.
        # Use a high default since Li.Fi's routing already accounts for depth.
        # The slippage_bps from toAmountMin is more reliable anyway.
        pool_liquidity_usd = 5_000_000.0

        route_id = (
            f"lifi_{chain_in.value}_{chain_out.value}_"
            f"{req.token_in}_{req.token_out}_{int(time.time() * 1000)}"
        )

        return RouteQuote(
            route_id=route_id,
            source=AggregatorSource.LIFI,
            route_type=route_type,
            chain_in=chain_in,
            chain_out=chain_out,
            token_in=req.token_in,
            token_out=req.token_out,
            amount_out=round(amount_out, 8),
            amount_out_min=round(amount_out_min, 8),
            slippage_bps=max(0, slippage_bps),
            gas_usd=round(gas_usd, 6),
            bridge_fee_usd=round(bridge_fee_usd, 6),
            bridge_time_seconds=bridge_time,
            pool_liquidity_usd=pool_liquidity_usd,
            price_impact_pct=round(slippage_bps / 100, 4),
            fetched_at_ms=int(time.time() * 1000),
        )

    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        logger.warning("Li.Fi | Failed to parse response: %s | keys=%s", exc, list(data.keys()))
        return None


# ---------------------------------------------------------------------------
# Route pair generator
# ---------------------------------------------------------------------------

def _route_pairs(chains: list[Chain]) -> list[tuple[Chain, Chain]]:
    """
    Generate (chain_in, chain_out) pairs to query.

    Strategy:
      - Always include same-chain routes (Li.Fi handles DEX-only swaps too).
      - Include all cross-chain combinations.
      - Exclude (A, A) duplicates only if the same pair would already be
        covered by the 1inch fetcher — but since Li.Fi uses its own routing
        engine, we include it for comparison.

    For N chains, this produces N² pairs.  With 3 chains → 9 pairs.
    """
    pairs = []
    for src in chains:
        for dst in chains:
            pairs.append((src, dst))
    return pairs


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_quotes(
    req: QuoteRequest,
    client: Optional[httpx.AsyncClient] = None,
) -> list[RouteQuote]:
    """
    Fetch Li.Fi quotes for all (chain_in, chain_out) route pairs in parallel.

    For N chains in req.chains this fires N² concurrent requests.
    With 3 chains (ethereum, arbitrum, base) → 9 concurrent requests.
    All failures are isolated — one pair failing never drops the others.

    Args:
        req:    The QuoteRequest from the agent.
        client: Optional shared httpx.AsyncClient.

    Returns:
        List of successfully parsed RouteQuote objects.  May be empty.
        Never raises.
    """
    pairs = _route_pairs(req.chains)

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )

    try:
        tasks = [
            _fetch_route(client, chain_in, chain_out, req)
            for chain_in, chain_out in pairs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[RouteQuote] = []
        for (chain_in, chain_out), result in zip(pairs, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Li.Fi | Unhandled exception %s→%s: %s",
                    chain_in.value, chain_out.value, result,
                )
            elif result is not None:
                quotes.append(result)

        logger.info(
            "Li.Fi | Fetched %d/%d quotes for %s→%s %.4f",
            len(quotes), len(pairs), req.token_in, req.token_out, req.amount_in,
        )
        return quotes

    finally:
        if own_client:
            await client.aclose()
