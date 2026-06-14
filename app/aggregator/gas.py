"""
aggregator/gas.py
-----------------
Fetches and normalises real-time gas prices for all supported chains.

Data sources (all free, no API key required):
  - Gas prices:  JSON-RPC eth_feeHistory against public llamarpc/L2 endpoints
  - ETH price:   CoinGecko simple/price (30 req/min free tier, no key)

Output: one GasSnapshot per chain, used by the optimizer to:
  1. Validate/cross-check the gas_usd values inside RouteQuotes.
  2. Attach a "gas_snapshot" block to the API response for agent auditability.

Design notes:
  - All chain fetches run in parallel (asyncio.gather).
  - eth_feeHistory gives us both base_fee (EIP-1559) and priority fee
    percentiles — much more accurate than the legacy eth_gasPrice call.
  - If any single chain RPC fails we return a None for that chain and
    continue — a missing gas snapshot never crashes the route scorer.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import httpx

from app.aggregator.constants import CHAIN_ID, PUBLIC_RPC
from app.optimizer.models import Chain, GasSnapshot
from app.optimizer.tec import build_gas_snapshot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of recent blocks to sample for fee history (more → smoother average)
FEE_HISTORY_BLOCKS = 5

# Priority fee percentile we target — 50th percentile is the median tip,
# safe for non-urgent execution.  Raise to 75 for faster inclusion.
PRIORITY_FEE_PERCENTILE = 50

# CoinGecko endpoint — no key required on free tier
COINGECKO_PRICE_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=ethereum&vs_currencies=usd"
)

# HTTP timeouts (seconds)
RPC_TIMEOUT   = 5.0
PRICE_TIMEOUT = 5.0

# Retry attempts for each individual call
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_eth_price(client: httpx.AsyncClient) -> float:
    """
    Fetch ETH/USD spot price from CoinGecko.
    Returns a fallback of 0.0 on failure (callers must check for this).
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.get(COINGECKO_PRICE_URL, timeout=PRICE_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return float(data["ethereum"]["usd"])
        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.warning("CoinGecko ETH price fetch failed after %d attempts: %s", MAX_RETRIES + 1, exc)
                return 0.0
            await asyncio.sleep(0.3 * (attempt + 1))  # 0.3s, 0.6s backoff
    return 0.0  # unreachable, satisfies type checker


async def _fetch_fee_history(
    client: httpx.AsyncClient,
    chain: Chain,
) -> Optional[dict]:
    """
    Call eth_feeHistory on the public RPC for one chain.

    JSON-RPC request structure:
        method:  eth_feeHistory
        params:  [block_count, "latest", [percentile]]
            block_count — number of recent blocks to average
            percentile  — which priority fee percentile to return

    Returns the raw JSON-RPC result dict, or None on failure.
    """
    rpc_url = PUBLIC_RPC[chain]
    payload = {
        "jsonrpc": "2.0",
        "method":  "eth_feeHistory",
        "params":  [FEE_HISTORY_BLOCKS, "latest", [PRIORITY_FEE_PERCENTILE]],
        "id":      CHAIN_ID[chain],
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.post(rpc_url, json=payload, timeout=RPC_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()

            if "error" in body:
                raise ValueError(f"RPC error: {body['error']}")

            return body["result"]

        except Exception as exc:
            if attempt == MAX_RETRIES:
                logger.warning(
                    "eth_feeHistory failed for %s after %d attempts: %s",
                    chain.value, MAX_RETRIES + 1, exc,
                )
                return None
            await asyncio.sleep(0.3 * (attempt + 1))

    return None


def _parse_fee_history(result: dict) -> tuple[float, float]:
    """
    Extract base_fee_gwei and priority_fee_gwei from an eth_feeHistory result.

    eth_feeHistory returns:
        baseFeePerGas: list of hex strings (one per block + one pending)
        reward:        list of [list of hex percentile values] per block

    We average the non-None values to smooth out single-block spikes.
    The last element of baseFeePerGas is the *pending* block's base fee —
    the most forward-looking value, so we weight it by using it directly
    rather than averaging with historical blocks.

    Returns: (base_fee_gwei, priority_fee_gwei) — both as floats in Gwei.
    """
    # Last entry in baseFeePerGas is the next (pending) block — use it directly
    raw_base_fees = result.get("baseFeePerGas", [])
    if not raw_base_fees:
        return 0.0, 0.0

    # Use the pending block base fee (last element)
    pending_base_fee_wei = int(raw_base_fees[-1], 16)
    base_fee_gwei = pending_base_fee_wei / 1e9

    # Average the priority fee across sampled blocks
    rewards = result.get("reward", [])  # [[hex_pct], [hex_pct], ...]
    priority_fees_wei = []
    for block_rewards in rewards:
        if block_rewards:
            priority_fees_wei.append(int(block_rewards[0], 16))

    if priority_fees_wei:
        priority_fee_gwei = (sum(priority_fees_wei) / len(priority_fees_wei)) / 1e9
    else:
        # Fallback: 1 Gwei tip is the minimum reasonable value
        priority_fee_gwei = 1.0

    return base_fee_gwei, priority_fee_gwei


async def _fetch_snapshot_for_chain(
    client: httpx.AsyncClient,
    chain: Chain,
    eth_price_usd: float,
) -> Optional[GasSnapshot]:
    """
    Fetch gas data for one chain and return a GasSnapshot.
    Returns None if the RPC call failed — callers filter out None values.
    """
    result = await _fetch_fee_history(client, chain)
    if result is None:
        return None

    base_fee_gwei, priority_fee_gwei = _parse_fee_history(result)

    if base_fee_gwei == 0.0:
        logger.warning("Chain %s returned zero base fee — RPC data may be stale", chain.value)

    snapshot = build_gas_snapshot(
        chain=chain,
        base_fee_gwei=base_fee_gwei,
        priority_fee_gwei=priority_fee_gwei,
        eth_price_usd=eth_price_usd,
    )

    logger.debug(
        "Gas snapshot | chain=%s base=%.2fgwei tip=%.2fgwei eth=$%.0f → $%.4f/unit",
        chain.value, base_fee_gwei, priority_fee_gwei, eth_price_usd,
        snapshot.gas_usd_per_unit,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

async def fetch_gas_snapshots(
    chains: list[Chain],
    client: Optional[httpx.AsyncClient] = None,
) -> dict[Chain, GasSnapshot]:
    """
    Fetch gas prices for all requested chains in parallel.

    Workflow:
      1. Fetch ETH/USD price once from CoinGecko (shared across all chains —
         all supported chains use ETH as the native gas token).
      2. Fan-out eth_feeHistory RPC calls to all chains simultaneously.
      3. Build GasSnapshot for each successful chain response.

    Args:
        chains: List of chains to fetch gas for (typically from QuoteRequest.chains).
        client: Optional shared httpx.AsyncClient (pass one in for connection reuse).
                If None, a fresh client is created and closed inside this function.

    Returns:
        Dict mapping Chain → GasSnapshot.  Chains whose RPC calls failed are
        absent from the dict — the caller must handle missing chains gracefully.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            headers={"Content-Type": "application/json"},
            follow_redirects=True,
        )

    try:
        # --- Step 1: ETH price (one call, shared) ---
        eth_price_usd = await _fetch_eth_price(client)
        if eth_price_usd == 0.0:
            logger.error(
                "ETH price is 0.0 — gas USD costs will be meaningless. "
                "CoinGecko may be down. Aborting gas fetch."
            )
            return {}

        # --- Step 2: Fan-out RPC calls ---
        tasks = [
            _fetch_snapshot_for_chain(client, chain, eth_price_usd)
            for chain in chains
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Step 3: Collect successful snapshots ---
        snapshots: dict[Chain, GasSnapshot] = {}
        for chain, result in zip(chains, results):
            if isinstance(result, Exception):
                logger.warning("Unexpected exception for chain %s: %s", chain.value, result)
            elif result is not None:
                snapshots[chain] = result

        logger.info(
            "Gas fetch complete | requested=%d successful=%d eth_price=$%.2f",
            len(chains), len(snapshots), eth_price_usd,
        )
        return snapshots

    finally:
        if own_client:
            await client.aclose()
