"""
optimizer/tec.py
----------------
True Execution Cost (TEC) algorithm and route selection logic.

Mathematical model:
    TEC  = C_slippage + C_gas + C_bridge + C_time_penalty
    NOV  = amount_out × target_price_usd
    SCORE = NOV - TEC - amount_in_usd          # net profit in USD

The optimizer is a pure function: given a list of RouteQuotes and a
QuoteRequest it returns a deterministic, sorted list of ScoredRoutes.
No I/O, no external calls, no randomness — fully unit-testable.

Zero LLM, zero heuristics: every number here is derived from
on-chain / aggregator data passed in through the RouteQuote contract.
"""

from __future__ import annotations

import time
from typing import Final, Optional

from .models import (
    Chain,
    GasSnapshot,
    QuoteRequest,
    RouteQuote,
    ScoredRoute,
    TECBreakdown,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Quotes older than this are stale and must not be scored.
# 3 seconds matches our Redis TTL in the aggregator layer.
QUOTE_STALENESS_MS: Final[int] = 3_000

# Pool impact model coefficients by AMM type.
# These map how aggressively price impact scales with order size relative
# to available liquidity.
#   Uniswap v3 concentrated: lower coefficient because liquidity is denser
#     near the current price.
#   Uniswap v2 / constant-product: full k=xy invariant impact.
#   Curve StableSwap: very tight peg, impact is minimal.
IMPACT_COEFF: Final[dict[str, float]] = {
    "uniswap_v3":   0.50,
    "uniswap_v2":   1.00,
    "curve":        0.05,
    "balancer":     0.75,
    "default":      0.80,   # conservative fallback
}

# Gas units consumed by a standard ERC-20 swap call.
# These are empirically measured medians — not aggregator estimates.
# Used only as a sanity-check upper bound; aggregator gas_usd takes
# precedence when provided.
GAS_UNITS_SWAP: Final[dict[Chain, int]] = {
    Chain.ETHEREUM: 150_000,
    Chain.ARBITRUM:  800_000,   # Arbitrum counts L2 gas differently
    Chain.BASE:      400_000,
    Chain.OPTIMISM:  400_000,
    Chain.POLYGON:   180_000,
}

# L1 calldata surcharge multipliers applied to raw gas_gwei × eth_price.
# Mainnet = 1.0 (no surcharge).  L2s are dramatically cheaper.
# Source: empirical medians from etherscan/arbiscan, 2025-Q1.
L1_SURCHARGE_MULTIPLIER: Final[dict[Chain, float]] = {
    Chain.ETHEREUM: 1.000,
    Chain.ARBITRUM: 0.010,   # ~100× cheaper than mainnet
    Chain.BASE:     0.005,   # ~200× cheaper
    Chain.OPTIMISM: 0.008,
    Chain.POLYGON:  0.002,
}


# ---------------------------------------------------------------------------
# Gas Normalisation
# ---------------------------------------------------------------------------

def normalise_gas_cost(
    chain: Chain,
    base_fee_gwei: float,
    priority_fee_gwei: float,
    eth_price_usd: float,
    gas_units: Optional[int] = None,
) -> float:
    """
    Convert raw gas parameters to a USD cost for one swap transaction.

    Formula:
        effective_gwei = base_fee_gwei + priority_fee_gwei
        gas_eth        = effective_gwei × 1e-9 × gas_units
        gas_usd_raw    = gas_eth × eth_price_usd
        gas_usd        = gas_usd_raw × L1_SURCHARGE_MULTIPLIER[chain]

    Args:
        chain:              Target chain.
        base_fee_gwei:      EIP-1559 base fee in Gwei.
        priority_fee_gwei:  Miner tip (priority fee) in Gwei.
        eth_price_usd:      Current ETH (or chain native token) price in USD.
        gas_units:          Override for gas units consumed.  Defaults to the
                            chain's empirical GAS_UNITS_SWAP constant.

    Returns:
        Estimated USD cost of the gas for this transaction.
    """
    units = gas_units if gas_units is not None else GAS_UNITS_SWAP[chain]
    effective_gwei = base_fee_gwei + priority_fee_gwei
    gas_eth = effective_gwei * 1e-9 * units
    gas_usd_raw = gas_eth * eth_price_usd
    return gas_usd_raw * L1_SURCHARGE_MULTIPLIER[chain]


def build_gas_snapshot(
    chain: Chain,
    base_fee_gwei: float,
    priority_fee_gwei: float,
    eth_price_usd: float,
) -> GasSnapshot:
    """
    Construct a GasSnapshot for one chain from raw parameters.
    Called by aggregator/gas.py before handing data to the optimizer.
    """
    multiplier = L1_SURCHARGE_MULTIPLIER[chain]
    effective_gwei = base_fee_gwei + priority_fee_gwei
    gas_usd_per_unit = effective_gwei * 1e-9 * eth_price_usd * multiplier

    return GasSnapshot(
        chain=chain,
        base_fee_gwei=base_fee_gwei,
        priority_fee_gwei=priority_fee_gwei,
        eth_price_usd=eth_price_usd,
        gas_usd_per_unit=gas_usd_per_unit,
        l1_surcharge_multiplier=multiplier,
    )


# ---------------------------------------------------------------------------
# Slippage Validation
# ---------------------------------------------------------------------------

def estimate_price_impact_bps(
    amount_in_usd: float,
    pool_liquidity_usd: float,
    amm_type: str = "default",
) -> int:
    """
    Independently estimate price impact in basis points from pool depth.

    This is our own calculation, cross-checked against the aggregator's
    reported slippage_bps.  If our estimate diverges significantly, we
    flag the route as potentially mis-quoted.

    Formula (generalised):
        impact_pct = (amount_in_usd / pool_liquidity_usd) × impact_coefficient
        impact_bps = impact_pct × 10_000

    Args:
        amount_in_usd:      Size of the trade in USD.
        pool_liquidity_usd: Total USD liquidity in the pool being used.
        amm_type:           AMM model key from IMPACT_COEFF.

    Returns:
        Estimated price impact in basis points.
    """
    if pool_liquidity_usd <= 0:
        return 10_000  # 100% impact — treat as unviable
    coeff = IMPACT_COEFF.get(amm_type, IMPACT_COEFF["default"])
    impact_pct = (amount_in_usd / pool_liquidity_usd) * coeff
    return int(impact_pct * 10_000)


def slippage_divergence_flag(
    quoted_bps: int,
    estimated_bps: int,
    tolerance_bps: int = 100,
) -> bool:
    """
    Return True if the aggregator's quoted slippage deviates from our
    pool-depth estimate by more than tolerance_bps.

    A large divergence suggests either stale liquidity data from the
    aggregator or a route that routes through a very thin pool that the
    aggregator is under-reporting.
    """
    return abs(quoted_bps - estimated_bps) > tolerance_bps


# ---------------------------------------------------------------------------
# Core TEC Computation
# ---------------------------------------------------------------------------

def compute_tec(
    quote: RouteQuote,
    req: QuoteRequest,
) -> tuple[TECBreakdown, float, float]:
    """
    Compute the True Execution Cost, Net Output Value, and Profit Score
    for a single RouteQuote.

    TEC Component Formulas:
    ───────────────────────
    C_slippage     = amount_in_usd × (slippage_bps / 10_000)

        Slippage is expressed as a fraction of the input value.  We use
        amount_in_usd rather than amount_out_usd because input size drives
        price impact — this makes the formula input-side consistent.

    C_gas          = quote.gas_usd
        Already normalised to USD by the aggregator using normalise_gas_cost().

    C_bridge       = quote.bridge_fee_usd
        Combined flat + percentage bridge fee, normalised by aggregator.

    C_time_penalty = bridge_time_seconds × agent_time_value_per_second
        Converts waiting time into a USD opportunity cost.  Single-chain
        routes have bridge_time_seconds=0, so this term vanishes.

    TEC = C_slippage + C_gas + C_bridge + C_time_penalty

    NOV (Net Output Value):
    ───────────────────────
    NOV = amount_out × target_price_usd
        The USD value the agent will actually receive.

    Profit Score:
    ─────────────
    SCORE = NOV - TEC - amount_in_usd
        Positive: the agent profits after all costs.
        Negative: the route costs more than it earns — never execute.
        Zero:     break-even — acceptable only if the agent is rebalancing.

    Args:
        quote:  Immutable RouteQuote from an aggregator.
        req:    The originating QuoteRequest (provides amount_in_usd,
                target_price_usd, agent_time_value_per_second).

    Returns:
        (TECBreakdown, net_output_value_usd, profit_score)
    """
    # --- C_slippage ---
    slippage_usd = req.amount_in_usd * (quote.slippage_bps / 10_000)

    # --- C_gas ---
    # Already in USD; no further computation needed.
    gas_usd = quote.gas_usd

    # --- C_bridge ---
    bridge_fee_usd = quote.bridge_fee_usd

    # --- C_time_penalty ---
    # Using integer seconds × float USD/s — no float precision concern
    # at this scale.
    time_penalty_usd = (
        quote.bridge_time_seconds * req.agent_time_value_per_second
    )

    breakdown = TECBreakdown(
        slippage_usd=slippage_usd,
        gas_usd=gas_usd,
        bridge_fee_usd=bridge_fee_usd,
        time_penalty_usd=time_penalty_usd,
    )

    # --- NOV ---
    nov = quote.amount_out * req.target_price_usd

    # --- Profit Score ---
    score = nov - breakdown.total - req.amount_in_usd

    return breakdown, nov, score


# ---------------------------------------------------------------------------
# Viability Checks  (hard constraints — fail any one → disqualify)
# ---------------------------------------------------------------------------

def _check_viability(
    quote: RouteQuote,
    req: QuoteRequest,
    now_ms: int,
) -> tuple[bool, Optional[str]]:
    """
    Apply hard constraints to a quote.  Returns (is_viable, reason).
    Any failing constraint immediately disqualifies the route.

    Hard constraints (in evaluation order):
      1. Staleness: quote must be fresh (within QUOTE_STALENESS_MS).
      2. Max slippage: quote.slippage_bps ≤ req.max_slippage_bps.
      3. Max bridge time: quote.bridge_time_seconds ≤ req.max_bridge_time_seconds.
      4. Positive output: amount_out_min must be > 0.
      5. Liquidity floor: pool_liquidity_usd must be > amount_in_usd × 5
         (i.e., order must be < 20% of pool depth to avoid extreme impact).
    """
    age_ms = now_ms - quote.fetched_at_ms
    if age_ms > QUOTE_STALENESS_MS:
        return False, f"Quote stale: {age_ms}ms old (limit {QUOTE_STALENESS_MS}ms)"

    if quote.slippage_bps > req.max_slippage_bps:
        return False, (
            f"Slippage {quote.slippage_bps}bps exceeds limit {req.max_slippage_bps}bps"
        )

    if quote.bridge_time_seconds > req.max_bridge_time_seconds:
        return False, (
            f"Bridge time {quote.bridge_time_seconds}s exceeds limit {req.max_bridge_time_seconds}s"
        )

    if quote.amount_out_min <= 0:
        return False, "amount_out_min is zero — route produces no output"

    liquidity_floor = req.amount_in_usd * 5.0
    if quote.pool_liquidity_usd < liquidity_floor:
        return False, (
            f"Pool liquidity ${quote.pool_liquidity_usd:,.0f} below 5× order size floor "
            f"${liquidity_floor:,.0f}"
        )

    return True, None


# ---------------------------------------------------------------------------
# Effective Rate
# ---------------------------------------------------------------------------

def _compute_effective_rate(quote: RouteQuote, amount_in: float) -> float:
    """
    Effective exchange rate: output token units per input token unit.
    Represents what the agent actually receives per unit sent, after all
    on-chain execution (slippage already factored into amount_out_min).

    Uses amount_out_min (the contract-enforced floor) rather than amount_out
    so agents see the worst-case rate, not the optimistic quoted rate.
    """
    if amount_in <= 0:
        return 0.0
    return quote.amount_out_min / amount_in


# ---------------------------------------------------------------------------
# Route Scorer — main entry point
# ---------------------------------------------------------------------------

def score_routes(
    quotes: list[RouteQuote],
    req: QuoteRequest,
    now_ms: Optional[int] = None,
) -> list[ScoredRoute]:
    """
    Score and rank all candidate RouteQuotes against a QuoteRequest.

    Processing pipeline (pure, no I/O):
      1. Validate staleness and hard constraints → mark is_viable.
      2. Compute TEC, NOV, and profit_score for every quote.
      3. Sort: viable routes by profit_score descending,
               then disqualified routes (for agent debugging).

    Args:
        quotes:   Flat list of RouteQuotes from all aggregators.
        req:      The originating QuoteRequest.
        now_ms:   Current time in ms (injectable for deterministic testing).

    Returns:
        Sorted list of ScoredRoutes, viable ones first.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    scored: list[ScoredRoute] = []

    for quote in quotes:
        is_viable, reason = _check_viability(quote, req, now_ms)

        breakdown, nov, score = compute_tec(quote, req)
        effective_rate = _compute_effective_rate(quote, req.amount_in)

        scored.append(ScoredRoute(
            quote=quote,
            true_execution_cost_usd=round(breakdown.total, 8),
            net_output_value_usd=round(nov, 8),
            profit_score=round(score, 8),
            cost_breakdown=breakdown,
            effective_rate=round(effective_rate, 8),
            is_viable=is_viable,
            disqualification_reason=reason,
        ))

    # Sort: viable routes first (profit_score desc), then disqualified routes.
    return sorted(
        scored,
        key=lambda sr: (sr.is_viable, sr.profit_score),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Optimal Route Selector
# ---------------------------------------------------------------------------

def select_optimal_route(scored: list[ScoredRoute]) -> Optional[ScoredRoute]:
    """
    Return the single best viable route from a pre-scored list.

    Optimality criterion: maximum profit_score among is_viable routes.
    The score_routes() sort guarantees the first viable route is already
    optimal, so this is an O(1) lookup.

    Returns None if no viable routes exist — the API layer must surface
    this as a 200 with optimal_route=null rather than a 4xx.  The agent
    should widen its constraints or retry with fresher quotes.
    """
    for sr in scored:
        if sr.is_viable:
            return sr
    return None


# ---------------------------------------------------------------------------
# Secondary Validation — Divergence Report
# ---------------------------------------------------------------------------

def slippage_audit(
    quotes: list[RouteQuote],
    req: QuoteRequest,
    amm_type: str = "default",
) -> list[dict]:
    """
    Cross-check every quote's reported slippage against our own
    pool-depth estimate.  Returns a report list for logging/monitoring.

    This does not affect route selection — it's an audit tool for
    detecting consistently mis-quoting aggregators so you can down-weight
    or blacklist them over time.

    Example output row:
        {
            "route_id": "lifi_ethereum_arbitrum_ETH_USDC_1718000000",
            "source": "lifi",
            "quoted_bps": 45,
            "estimated_bps": 62,
            "diverged": True,
        }
    """
    report = []
    for q in quotes:
        estimated = estimate_price_impact_bps(
            amount_in_usd=req.amount_in_usd,
            pool_liquidity_usd=q.pool_liquidity_usd,
            amm_type=amm_type,
        )
        diverged = slippage_divergence_flag(q.slippage_bps, estimated)
        report.append({
            "route_id":      q.route_id,
            "source":        q.source.value,
            "quoted_bps":    q.slippage_bps,
            "estimated_bps": estimated,
            "diverged":      diverged,
        })
    return report
