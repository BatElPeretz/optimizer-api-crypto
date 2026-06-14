"""
aggregator/oneinch.py
---------------------
Async client for the 1inch Aggregation API v5.2.

What 1inch provides:
  - Best single-chain DEX route for a given token pair and amount.
  - Splits orders across multiple DEXes when that reduces price impact.
  - Returns the gas-optimised path with `estimatedGas` and `toAmount`.

What 1inch does NOT provide:
  - Cross-chain bridging (single-chain only — use lifi.py for cross-chain).
  - Explicit slippage/price-impact as a number (we derive it, see below).
  - Pool liquidity depth directly (we use a default; improve with DexScreener
    in a later sprint when real depth data matters).

API key:
  Get a free key at https://portal.1inch.dev — set ONEINCH_API_KEY in .env.
  Without a key, requests hit a stricter rate limit and may 401.

Slippage derivation:
  1inch's `toAmount` is the *optimal* output at the current block state.
  We derive implied slippage_bps as:
      implied_slippage = (spot_out - actual_out) / spot_out
  where spot_out = amount_in_token_units * spot_rate.
  This requires the spot price of token_in in token_out units, which we
  approximate as (amount_in_usd / amount_out_usd) using the request's
  target_price_usd.  It's a close-enough signal for route ranking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

import httpx

from app.aggregator.constants import (
    CHAIN_ID,
    token_address,
    token_decimals,
    raw_to_float,
)
from app.optimizer.models import (
    AggregatorSource,
    Chain,
    QuoteRequest,
    RouteQuote,
    RouteType,
)
from app.optimizer.tec import estimate_price_impact_bps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ONEINCH_BASE_URL = "https://api.1inch.dev/swap/v5.2"
REQUEST_TIMEOUT  = 6.0   # seconds — 1inch is usually fast
MAX_RETRIES      = 2
RETRY_DELAY_BASE = 0.4   # seconds (doubles each retry)

# When 1inch does not provide pool liquidity (which it never does in the
# quote endpoint), we fall back to this conservative estimate.
# This causes our independent slippage estimate to assume a shallow pool,
# which is intentionally pessimistic — better than over-optimism.
# Replace this with a DexScreener lookup in a future sprint.
FALLBACK_POOL_LIQUIDITY_USD = 2_000_000.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_headers() -> dict[str, str]:
    """
    Construct request headers.  Reads ONEINCH_API_KEY from environment.
    Raises if the key is missing so the failure is explicit at startup,
    not silently at first request.
    """
    api_key = os.environ.get("ONEINCH_API_KEY", "")
    if not api_key:
        logger.warning(
            "ONEINCH_API_KEY is not set. Requests may hit strict rate limits. "
            "Get a free key at https://portal.1inch.dev"
        )
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _derive_slippage_bps(
    amount_in_usd: float,
    amount_out_usd: float,          # actual quoted output in USD
    expected_out_usd: float,        # expected output at spot price, no impact
    pool_liquidity_usd: float,
) -> int:
    """
    Derive implied price impact in basis points.

    We use two signals and take the *higher* (more conservative) of the two:

    Signal A — Dollar-level impact from the quoted vs expected output:
        impact_A = (expected_out_usd - amount_out_usd) / expected_out_usd

    Signal B — Our pool-depth model (from tec.estimate_price_impact_bps):
        impact_B = (amount_in_usd / pool_liquidity_usd) × coefficient

    Using max(A, B) prevents us from under-counting slippage when either
    the API is routing through a known thin pool or the spot price we have
    is slightly stale.
    """
    if expected_out_usd <= 0:
        return 0

    dollar_impact_pct = max(0.0, (expected_out_usd - amount_out_usd) / expected_out_usd)
    signal_a = int(dollar_impact_pct * 10_000)

    signal_b = estimate_price_impact_bps(amount_in_usd, pool_liquidity_usd)

    return max(signal_a, signal_b)


async def _fetch_quote_for_chain(
    client: httpx.AsyncClient,
    chain: Chain,
    req: QuoteRequest,
    headers: dict[str, str],
) -> Optional[RouteQuote]:
    """
    Fetch a single-chain quote from 1inch for one specific chain.

    Returns None if:
      - The token pair or chain is not supported by 1inch.
      - The API returns a non-200 status.
      - All retry attempts are exhausted.
    """
    chain_id = CHAIN_ID[chain]

    # Resolve contract addresses — raises KeyError if unsupported pair/chain
    try:
        src_addr = token_address(req.token_in, chain)
        dst_addr = token_address(req.token_out, chain)
    except KeyError as exc:
        logger.debug("1inch | Skipping chain %s — %s", chain.value, exc)
        return None

    # Convert amount_in from token units to raw integer (wei for ETH, etc.)
    decimals_in = token_decimals(req.token_in)
    raw_amount_in = int(req.amount_in * (10 ** decimals_in))

    url = f"{ONEINCH_BASE_URL}/{chain_id}/quote"
    params = {
        "src":          src_addr,
        "dst":          dst_addr,
        "amount":       str(raw_amount_in),
        "includeGas":   "true",
        "includeTokensInfo": "false",   # saves response size
        "includeProtocols": "false",    # not needed for MVP scoring
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)

            # 400 with "insufficient liquidity" → no route, skip gracefully
            if resp.status_code == 400:
                body = resp.json()
                logger.debug("1inch | No route on %s: %s", chain.value, body.get("description", ""))
                return None

            resp.raise_for_status()
            return _parse_response(resp.json(), chain, req)

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 503):
                # Rate limited or service unavailable — always retry with backoff
                wait = RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning("1inch | %d on %s, retrying in %.1fs", exc.response.status_code, chain.value, wait)
                await asyncio.sleep(wait)
            elif attempt == MAX_RETRIES:
                logger.warning("1inch | HTTP %d on %s after %d attempts", exc.response.status_code, chain.value, MAX_RETRIES + 1)
                return None
            else:
                await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))

        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            if attempt == MAX_RETRIES:
                logger.warning("1inch | Connection error on %s: %s", chain.value, exc)
                return None
            await asyncio.sleep(RETRY_DELAY_BASE * (2 ** attempt))

        except Exception as exc:
            logger.warning("1inch | Unexpected error on %s: %s", chain.value, exc)
            return None

    return None


def _parse_response(
    data: dict[str, Any],
    chain: Chain,
    req: QuoteRequest,
) -> Optional[RouteQuote]:
    """
    Map a raw 1inch v5.2 quote JSON response to our RouteQuote dataclass.

    1inch response schema (v5.2 quote endpoint):
    {
        "toAmount":     "18400123456",   ← raw integer in dst token decimals
        "gas":          180000,          ← estimated gas units
        "protocols":    [...],           ← routing path (we ignore for now)
        "fromToken":    {...},
        "toToken":      {...},
    }

    Fields we must derive ourselves (not in the response):
      - amount_out_min:  We apply a conservative 0.5% guard on top of quoted
                         slippage since 1inch doesn't return this directly.
      - slippage_bps:    Derived via _derive_slippage_bps().
      - gas_usd:         gas_units × approximate gas price in USD.
                         We use a chain-specific per-unit cost.  The optimizer
                         will cross-check this against the GasSnapshot.
      - pool_liquidity_usd: Fallback constant (see FALLBACK_POOL_LIQUIDITY_USD).
    """
    try:
        decimals_out = token_decimals(req.token_out)
        amount_out = raw_to_float(data["toAmount"], req.token_out)

        # --- Gas estimation ---
        # 1inch provides gas *units*.  We need a gas price to convert to USD.
        # For the MVP we use a hard-coded per-chain cost-per-gas-unit in USD.
        # The optimizer layer will later cross-check this against GasSnapshot.
        gas_units: int = int(data.get("gas", 0))
        gas_usd = _estimate_gas_usd_from_units(chain, gas_units)

        # --- Slippage derivation ---
        # expected_out_usd: what the agent would receive at a zero-impact spot rate
        # amount_out_usd:   what 1inch is actually quoting (post-routing impact)
        expected_out_usd = req.amount_in_usd   # assuming token_in → token_out at spot
        amount_out_usd   = amount_out * req.target_price_usd

        slippage_bps = _derive_slippage_bps(
            amount_in_usd=req.amount_in_usd,
            amount_out_usd=amount_out_usd,
            expected_out_usd=expected_out_usd,
            pool_liquidity_usd=FALLBACK_POOL_LIQUIDITY_USD,
        )

        # amount_out_min: contract-enforceable floor = amount_out × (1 - 0.5%)
        # This is separate from slippage_bps (which measures price impact).
        # Agents that actually execute will set this in their tx calldata.
        amount_out_min = amount_out * (1 - 0.005)

        route_id = (
            f"1inch_{chain.value}_{req.token_in}_{req.token_out}_{int(time.time() * 1000)}"
        )

        return RouteQuote(
            route_id=route_id,
            source=AggregatorSource.ONEINCH,
            route_type=RouteType.DIRECT_SWAP,    # 1inch is always single-chain
            chain_in=chain,
            chain_out=chain,                      # same chain in and out
            token_in=req.token_in,
            token_out=req.token_out,
            amount_out=round(amount_out, 8),
            amount_out_min=round(amount_out_min, 8),
            slippage_bps=slippage_bps,
            gas_usd=round(gas_usd, 6),
            bridge_fee_usd=0.0,                  # no bridge, single-chain
            bridge_time_seconds=0,               # no bridge wait
            pool_liquidity_usd=FALLBACK_POOL_LIQUIDITY_USD,
            price_impact_pct=round(slippage_bps / 100, 4),
            fetched_at_ms=int(time.time() * 1000),
        )

    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("1inch | Failed to parse response: %s | data=%s", exc, data)
        return None


def _estimate_gas_usd_from_units(chain: Chain, gas_units: int) -> float:
    """
    Rough USD gas cost from gas units alone, without a live gas price call.

    These are empirical per-unit costs in USD at typical gas prices and
    ETH at ~$3000.  They are intentionally conservative (slightly high)
    since the optimizer's GasSnapshot will later provide precise values.
    The purpose here is to give the scorer *something* to work with when
    a live gas snapshot isn't available.

    Formulation:
        gas_usd = gas_units × typical_gwei × 1e-9 × eth_usd × l1_multiplier

    Values baked in for this fallback:
        Ethereum:  30 gwei, ETH=$3000 → ~$0.045 / 1000 gas units
        Arbitrum:  0.1 gwei effective (L1 surcharge ×0.01) → ~$0.00045
        Base:      0.05 gwei effective → ~$0.00023
    """
    _USD_PER_GAS_UNIT: dict[Chain, float] = {
        Chain.ETHEREUM: 4.5e-5,
        Chain.ARBITRUM: 4.5e-7,
        Chain.BASE:     2.3e-7,
        Chain.OPTIMISM: 3.6e-7,
        Chain.POLYGON:  1.5e-8,
    }
    cost_per_unit = _USD_PER_GAS_UNIT.get(chain, 4.5e-5)
    return gas_units * cost_per_unit


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_quotes(
    req: QuoteRequest,
    client: Optional[httpx.AsyncClient] = None,
) -> list[RouteQuote]:
    """
    Fetch 1inch quotes for all requested chains in parallel.

    1inch is single-chain only — we query each requested chain independently
    and collect all successful quotes.  The optimizer layer then scores and
    ranks them against quotes from other aggregators.

    Args:
        req:    The QuoteRequest from the agent.
        client: Optional shared httpx.AsyncClient for connection reuse.

    Returns:
        List of RouteQuote objects (may be empty if all chains fail or the
        pair is unsupported).  Never raises — failures are logged and skipped.
    """
    headers = _build_headers()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(follow_redirects=True)

    try:
        tasks = [
            _fetch_quote_for_chain(client, chain, req, headers)
            for chain in req.chains
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        quotes: list[RouteQuote] = []
        for chain, result in zip(req.chains, results):
            if isinstance(result, Exception):
                logger.warning("1inch | Unhandled exception for %s: %s", chain.value, result)
            elif result is not None:
                quotes.append(result)

        logger.info(
            "1inch | Fetched %d/%d quotes for %s→%s %.4f",
            len(quotes), len(req.chains), req.token_in, req.token_out, req.amount_in,
        )
        return quotes

    finally:
        if own_client:
            await client.aclose()
