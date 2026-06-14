"""
optimizer/models.py
-------------------
All Pydantic schemas and core data structures for the Cross-Chain
Liquidity & Execution Route Optimizer.

Design principles:
  - Zero LLM at runtime: every field is a concrete numeric or enum value.
  - Strict typing throughout — Pydantic v2 with model_config for performance.
  - RouteQuote is the single interface contract between the aggregator
    layer (Shay's track) and the optimizer layer (your track). Both sides
    must agree on this shape and never change it unilaterally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Chain(str, Enum):
    """Supported EVM-compatible chains.  Add new chains here only — do not
    scatter chain IDs as magic strings throughout the codebase."""
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    BASE     = "base"
    OPTIMISM = "optimism"
    POLYGON  = "polygon"


class AggregatorSource(str, Enum):
    """Which aggregator produced a given RouteQuote."""
    ONEINCH = "1inch"
    LIFI    = "lifi"
    SOCKET  = "socket"


class RouteType(str, Enum):
    """High-level classification of a route's execution path."""
    DIRECT_SWAP   = "direct_swap"    # single-chain, single DEX hop
    MULTI_HOP     = "multi_hop"      # single-chain, multiple DEX hops
    BRIDGE_THEN_SWAP = "bridge_then_swap"  # cross-chain: bridge first, then swap
    SPLIT_ROUTE   = "split_route"    # order split across multiple DEXes/routes


# ---------------------------------------------------------------------------
# Core Data Contract — RouteQuote
# ---------------------------------------------------------------------------
# This dataclass is the *only* object Shay's aggregators hand to the
# optimizer.  Using a dataclass (not Pydantic) here is intentional: it
# removes serialization overhead on the hot path; Pydantic is reserved for
# the HTTP boundary (request/response).

@dataclass(frozen=True)
class RouteQuote:
    """
    Immutable snapshot of a single route quote from one aggregator.

    All monetary values are in USD unless the field name says otherwise.
    All fee values must be >= 0.  The aggregator layer is responsible for
    normalising raw API responses into this schema — the optimizer must
    never parse raw aggregator JSON.
    """

    # --- Identity ---
    route_id: str
    """Stable unique ID: f'{source.value}_{chain_in}_{chain_out}_{token_in}_{token_out}_{int(time.time())}'"""

    source: AggregatorSource
    """Which aggregator produced this quote."""

    route_type: RouteType
    """High-level route classification for reporting purposes."""

    # --- Chains & Tokens ---
    chain_in: Chain
    """Chain where the input token originates."""

    chain_out: Chain
    """Chain where the output token will land."""

    token_in: str
    """Input token symbol, e.g. 'ETH'."""

    token_out: str
    """Output token symbol, e.g. 'USDC'."""

    # --- Output ---
    amount_out: float
    """Quoted output amount in target token units *before* slippage guard."""

    amount_out_min: float
    """Minimum output amount after slippage guard (what the contract enforces)."""

    # --- Cost Components (all in USD) ---
    slippage_bps: int
    """Price impact in basis points (1 bps = 0.01%).  Must be >= 0."""

    gas_usd: float
    """Estimated on-chain gas cost in USD.  Includes L1 calldata cost for
    L2s (Arbitrum, Base, Optimism) where applicable."""

    bridge_fee_usd: float
    """Total bridge fee in USD (flat + percentage components combined).
    Zero for single-chain routes."""

    bridge_time_seconds: int
    """Expected cross-chain settlement time in seconds.
    Zero for single-chain routes."""

    # --- Liquidity Quality Signals ---
    pool_liquidity_usd: float
    """Total USD liquidity in the deepest pool on this route.  Used for
    secondary slippage validation independent of the quoted slippage_bps."""

    price_impact_pct: float
    """Aggregator-reported price impact as a percentage (0.0 – 100.0).
    Cross-checked against our own slippage_bps computation."""

    # --- Metadata ---
    fetched_at_ms: int
    """Unix timestamp in milliseconds when this quote was fetched.
    Quotes older than QUOTE_STALENESS_MS must be discarded by the scorer."""

    def __post_init__(self) -> None:
        # Invariant guards — fail fast rather than silently scoring bad data.
        assert self.slippage_bps >= 0,       "slippage_bps must be non-negative"
        assert self.gas_usd >= 0,             "gas_usd must be non-negative"
        assert self.bridge_fee_usd >= 0,      "bridge_fee_usd must be non-negative"
        assert self.bridge_time_seconds >= 0, "bridge_time_seconds must be non-negative"
        assert self.pool_liquidity_usd > 0,   "pool_liquidity_usd must be positive"
        assert self.amount_out >= 0,          "amount_out must be non-negative"
        assert self.amount_out_min >= 0,      "amount_out_min must be non-negative"
        assert self.amount_out_min <= self.amount_out, (
            "amount_out_min cannot exceed amount_out"
        )


# ---------------------------------------------------------------------------
# HTTP Request Schema
# ---------------------------------------------------------------------------

class QuoteRequest(BaseModel):
    """
    Agent-facing POST /v1/quote request body.

    Agents must supply pre-converted USD values so the optimizer never
    needs to hit a price oracle mid-request — keeping latency minimal.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # --- What to swap ---
    token_in: Annotated[str, Field(
        min_length=1, max_length=10,
        description="Input token ticker symbol, e.g. 'ETH'",
        examples=["ETH"],
    )]

    token_out: Annotated[str, Field(
        min_length=1, max_length=10,
        description="Output token ticker symbol, e.g. 'USDC'",
        examples=["USDC"],
    )]

    amount_in: Annotated[float, Field(
        gt=0,
        description="Input amount in token units, e.g. 5.0 for 5 ETH",
        examples=[5.0],
    )]

    amount_in_usd: Annotated[float, Field(
        gt=0,
        description="USD value of amount_in at request time (agent pre-computes this)",
        examples=[18500.0],
    )]

    target_price_usd: Annotated[float, Field(
        gt=0,
        description="Current spot price of token_out in USD",
        examples=[1.0],
    )]

    chains: Annotated[list[Chain], Field(
        min_length=1, max_length=5,
        description="Chains to include in route search",
        examples=[["ethereum", "arbitrum", "base"]],
    )]

    # --- Constraints ---
    max_slippage_bps: Annotated[int, Field(
        default=100, ge=1, le=2000,
        description="Maximum acceptable slippage in basis points (default 100 = 1%)",
    )]

    agent_time_value_per_second: Annotated[float, Field(
        default=0.0001, ge=0.0,
        description=(
            "USD value of one second of waiting time for the calling agent. "
            "Used in the time penalty component of TEC. "
            "Default $0.36/hr — tune per agent SLA."
        ),
    )]

    max_bridge_time_seconds: Annotated[int, Field(
        default=900, ge=0,
        description="Hard upper limit on acceptable bridge time (seconds). Routes exceeding this are excluded.",
    )]

    @field_validator("chains")
    @classmethod
    def chains_must_be_unique(cls, v: list[Chain]) -> list[Chain]:
        if len(v) != len(set(v)):
            raise ValueError("chains list must not contain duplicates")
        return v

    @model_validator(mode="after")
    def token_in_out_must_differ(self) -> "QuoteRequest":
        if self.token_in.upper() == self.token_out.upper():
            raise ValueError("token_in and token_out must be different")
        return self


# ---------------------------------------------------------------------------
# Internal Computation Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TECBreakdown:
    """
    Itemised cost components that sum to True Execution Cost.
    Stored alongside every scored route for agent transparency.
    """
    slippage_usd: float
    gas_usd: float
    bridge_fee_usd: float
    time_penalty_usd: float

    @property
    def total(self) -> float:
        """Sum of all cost components in USD."""
        return self.slippage_usd + self.gas_usd + self.bridge_fee_usd + self.time_penalty_usd


@dataclass(frozen=True)
class ScoredRoute:
    """
    A RouteQuote annotated with its optimizer outputs.
    This is what the scorer produces; one per RouteQuote candidate.
    """
    quote: RouteQuote

    # --- TEC outputs ---
    true_execution_cost_usd: float
    """Sum of all cost components (TEC = slippage + gas + bridge fee + time penalty)."""

    net_output_value_usd: float
    """Expected value received: amount_out × target_price_usd."""

    profit_score: float
    """net_output_value_usd - true_execution_cost_usd - amount_in_usd.
    Higher is better.  The optimal route is argmax(profit_score)."""

    cost_breakdown: TECBreakdown
    """Itemised cost components for agent introspection."""

    effective_rate: float
    """amount_out / amount_in — the actual exchange rate after all costs."""

    is_viable: bool
    """False if the route violates any hard constraint (slippage, bridge time, negative output)."""

    disqualification_reason: Optional[str]
    """Human-readable reason if is_viable is False, else None."""


# ---------------------------------------------------------------------------
# Gas Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GasSnapshot:
    """
    Current gas prices for all requested chains, normalised to USD.
    Produced by aggregator/gas.py and attached to the response for
    agent auditability.
    """
    chain: Chain
    base_fee_gwei: float
    priority_fee_gwei: float
    eth_price_usd: float        # native gas token price (ETH on all supported chains)
    gas_usd_per_unit: float     # base_fee_gwei × 1e-9 × eth_price_usd
    l1_surcharge_multiplier: float
    """1.0 for Ethereum mainnet.  ~0.01 for Arbitrum.  ~0.005 for Base.
    Multiply gas_usd_per_unit by this to get true L2 gas cost."""


# ---------------------------------------------------------------------------
# HTTP Response Schema
# ---------------------------------------------------------------------------

class RouteBreakdownResponse(BaseModel):
    """Per-route detail surfaced in the API response."""

    model_config = {"frozen": True}

    route_id: str
    source: str
    route_type: str
    chain_in: str
    chain_out: str
    token_in: str
    token_out: str

    # Outputs
    amount_out: float
    amount_out_min: float
    effective_rate: float

    # Scores
    profit_score: float
    net_output_value_usd: float
    true_execution_cost_usd: float

    # Cost breakdown
    slippage_usd: float
    gas_usd: float
    bridge_fee_usd: float
    time_penalty_usd: float

    # Quality signals
    slippage_bps: int
    pool_liquidity_usd: float
    bridge_time_seconds: int

    # Viability
    is_viable: bool
    disqualification_reason: Optional[str]

    @classmethod
    def from_scored(cls, sr: ScoredRoute) -> "RouteBreakdownResponse":
        q = sr.quote
        b = sr.cost_breakdown
        return cls(
            route_id=q.route_id,
            source=q.source.value,
            route_type=q.route_type.value,
            chain_in=q.chain_in.value,
            chain_out=q.chain_out.value,
            token_in=q.token_in,
            token_out=q.token_out,
            amount_out=q.amount_out,
            amount_out_min=q.amount_out_min,
            effective_rate=sr.effective_rate,
            profit_score=round(sr.profit_score, 8),
            net_output_value_usd=round(sr.net_output_value_usd, 6),
            true_execution_cost_usd=round(sr.true_execution_cost_usd, 6),
            slippage_usd=round(b.slippage_usd, 6),
            gas_usd=round(b.gas_usd, 6),
            bridge_fee_usd=round(b.bridge_fee_usd, 6),
            time_penalty_usd=round(b.time_penalty_usd, 6),
            slippage_bps=q.slippage_bps,
            pool_liquidity_usd=q.pool_liquidity_usd,
            bridge_time_seconds=q.bridge_time_seconds,
            is_viable=sr.is_viable,
            disqualification_reason=sr.disqualification_reason,
        )


class OptimalRouteResponse(BaseModel):
    """
    Top-level API response for POST /v1/quote.

    Structure:
      optimal_route  — the single best route (highest profit_score among viable routes)
      all_routes     — full ranked list (viable first, then disqualified, for agent debugging)
      gas_snapshot   — current gas prices per chain for agent auditability
      meta           — request echo + timing stats
    """

    model_config = {"frozen": True}

    optimal_route: Optional[RouteBreakdownResponse]
    """None only if zero viable routes were found — agent should retry or widen constraints."""

    all_routes: list[RouteBreakdownResponse]
    """Sorted descending by profit_score.  Viable routes first, then disqualified ones."""

    gas_snapshot: list[dict]
    """Gas prices per chain at time of query."""

    meta: dict
    """Request echo (token_in, token_out, amount_in, chains) + latency_ms + quote_count."""
