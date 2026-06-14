from typing import Optional
"""
tests/test_tec.py
-----------------
Unit tests for the TEC algorithm and route scoring logic.
All tests are pure — no I/O, no network, no external dependencies.
Run with:  pytest tests/test_tec.py -v
"""

import time
import pytest

from app.optimizer.models import (
    AggregatorSource, Chain, QuoteRequest, RouteQuote, RouteType,
)
from app.optimizer.tec import (
    QUOTE_STALENESS_MS,
    compute_tec,
    estimate_price_impact_bps,
    normalise_gas_cost,
    score_routes,
    select_optimal_route,
    slippage_divergence_flag,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW_MS = int(time.time() * 1000)

def make_quote(
    route_id: str = "test_route",
    source: AggregatorSource = AggregatorSource.ONEINCH,
    route_type: RouteType = RouteType.DIRECT_SWAP,
    chain_in: Chain = Chain.ARBITRUM,
    chain_out: Chain = Chain.ARBITRUM,
    token_in: str = "ETH",
    token_out: str = "USDC",
    amount_out: float = 18_400.0,
    amount_out_min: float = 18_216.0,
    slippage_bps: int = 50,
    gas_usd: float = 0.30,
    bridge_fee_usd: float = 0.0,
    bridge_time_seconds: int = 0,
    pool_liquidity_usd: float = 10_000_000.0,
    price_impact_pct: float = 0.05,
    fetched_at_ms: Optional[int] = None,
) -> RouteQuote:
    return RouteQuote(
        route_id=route_id,
        source=source,
        route_type=route_type,
        chain_in=chain_in,
        chain_out=chain_out,
        token_in=token_in,
        token_out=token_out,
        amount_out=amount_out,
        amount_out_min=amount_out_min,
        slippage_bps=slippage_bps,
        gas_usd=gas_usd,
        bridge_fee_usd=bridge_fee_usd,
        bridge_time_seconds=bridge_time_seconds,
        pool_liquidity_usd=pool_liquidity_usd,
        price_impact_pct=price_impact_pct,
        fetched_at_ms=fetched_at_ms if fetched_at_ms is not None else NOW_MS,
    )


def make_request(
    amount_in: float = 5.0,
    amount_in_usd: float = 18_500.0,
    target_price_usd: float = 1.0,
    chains: Optional[list[Chain]] = None,
    max_slippage_bps: int = 100,
    agent_time_value_per_second: float = 0.0001,
    max_bridge_time_seconds: int = 900,
) -> QuoteRequest:
    return QuoteRequest(
        token_in="ETH",
        token_out="USDC",
        amount_in=amount_in,
        amount_in_usd=amount_in_usd,
        target_price_usd=target_price_usd,
        chains=chains or [Chain.ARBITRUM],
        max_slippage_bps=max_slippage_bps,
        agent_time_value_per_second=agent_time_value_per_second,
        max_bridge_time_seconds=max_bridge_time_seconds,
    )


# ---------------------------------------------------------------------------
# Gas normalisation
# ---------------------------------------------------------------------------

class TestNormaliseGasCost:
    def test_mainnet_gas(self):
        # 30 gwei base + 2 gwei tip, ETH=$3000, 150k units on Ethereum
        cost = normalise_gas_cost(
            chain=Chain.ETHEREUM,
            base_fee_gwei=30.0,
            priority_fee_gwei=2.0,
            eth_price_usd=3_000.0,
            gas_units=150_000,
        )
        # (30+2) × 1e-9 × 150000 × 3000 × 1.0 = 0.0144 ETH × 3000 = $14.40
        assert abs(cost - 14.40) < 0.01

    def test_arbitrum_is_much_cheaper(self):
        mainnet = normalise_gas_cost(Chain.ETHEREUM, 30.0, 2.0, 3_000.0, 150_000)
        arbitrum = normalise_gas_cost(Chain.ARBITRUM, 30.0, 2.0, 3_000.0, 150_000)
        # Arbitrum multiplier is 0.01 — 100× cheaper for same gas params
        assert arbitrum < mainnet / 50

    def test_zero_base_fee_returns_only_tip_cost(self):
        cost = normalise_gas_cost(Chain.BASE, 0.0, 1.0, 2_000.0, 400_000)
        expected = 1e-9 * 400_000 * 2_000.0 * 0.005
        assert abs(cost - expected) < 1e-9


# ---------------------------------------------------------------------------
# TEC formula
# ---------------------------------------------------------------------------

class TestComputeTEC:
    def test_single_chain_no_bridge(self):
        quote = make_quote(
            slippage_bps=50,        # 0.5%
            gas_usd=0.30,
            bridge_fee_usd=0.0,
            bridge_time_seconds=0,
        )
        req = make_request(amount_in_usd=18_500.0, target_price_usd=1.0)
        breakdown, nov, score = compute_tec(quote, req)

        # C_slippage = 18500 × 0.005 = 92.50
        assert abs(breakdown.slippage_usd - 92.50) < 0.001
        # C_gas = 0.30
        assert abs(breakdown.gas_usd - 0.30) < 0.001
        # C_bridge = 0.0
        assert breakdown.bridge_fee_usd == 0.0
        # C_time = 0 × 0.0001 = 0.0
        assert breakdown.time_penalty_usd == 0.0
        # TEC = 92.50 + 0.30 = 92.80
        assert abs(breakdown.total - 92.80) < 0.001
        # NOV = 18400 × 1.0 = 18400
        assert abs(nov - 18_400.0) < 0.001
        # SCORE = 18400 - 92.80 - 18500 = -192.80
        assert abs(score - (-192.80)) < 0.001

    def test_cross_chain_bridge_time_penalty(self):
        quote = make_quote(
            bridge_fee_usd=5.0,
            bridge_time_seconds=600,    # 10 minutes
        )
        req = make_request(agent_time_value_per_second=0.01)  # $36/hr agent
        breakdown, _, _ = compute_tec(quote, req)

        # C_time = 600 × 0.01 = 6.0
        assert abs(breakdown.time_penalty_usd - 6.0) < 0.001
        assert abs(breakdown.bridge_fee_usd - 5.0) < 0.001

    def test_zero_slippage_route(self):
        quote = make_quote(slippage_bps=0, gas_usd=0.0, bridge_fee_usd=0.0)
        req = make_request(amount_in_usd=1_000.0)
        breakdown, _, _ = compute_tec(quote, req)
        assert breakdown.slippage_usd == 0.0

    def test_tec_total_is_sum_of_components(self):
        quote = make_quote(slippage_bps=80, gas_usd=1.50, bridge_fee_usd=3.0,
                           bridge_time_seconds=300)
        req = make_request(amount_in_usd=5_000.0, agent_time_value_per_second=0.005)
        breakdown, _, _ = compute_tec(quote, req)
        expected_total = (
            breakdown.slippage_usd
            + breakdown.gas_usd
            + breakdown.bridge_fee_usd
            + breakdown.time_penalty_usd
        )
        assert abs(breakdown.total - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# Slippage estimation
# ---------------------------------------------------------------------------

class TestSlippageEstimation:
    def test_large_order_thin_pool_high_impact(self):
        bps = estimate_price_impact_bps(
            amount_in_usd=100_000.0,
            pool_liquidity_usd=200_000.0,
        )
        # 100k / 200k × 0.80 × 10000 = 4000 bps (40%)
        assert bps == 4_000

    def test_small_order_deep_pool_low_impact(self):
        bps = estimate_price_impact_bps(
            amount_in_usd=100.0,
            pool_liquidity_usd=10_000_000.0,
        )
        assert bps < 10  # < 0.1 bps

    def test_zero_liquidity_returns_max_impact(self):
        bps = estimate_price_impact_bps(100.0, 0.0)
        assert bps == 10_000  # 100%

    def test_v3_lower_impact_than_v2(self):
        kwargs = dict(amount_in_usd=10_000.0, pool_liquidity_usd=1_000_000.0)
        v3 = estimate_price_impact_bps(**kwargs, amm_type="uniswap_v3")
        v2 = estimate_price_impact_bps(**kwargs, amm_type="uniswap_v2")
        assert v3 < v2

    def test_divergence_flag_triggers_when_spread_exceeds_tolerance(self):
        assert slippage_divergence_flag(50, 200, tolerance_bps=100) is True

    def test_divergence_flag_clear_when_spread_within_tolerance(self):
        assert slippage_divergence_flag(50, 80, tolerance_bps=100) is False


# ---------------------------------------------------------------------------
# Viability checks & scorer
# ---------------------------------------------------------------------------

class TestScoreRoutes:
    def test_stale_quote_is_disqualified(self):
        stale_ms = NOW_MS - QUOTE_STALENESS_MS - 1
        quote = make_quote(fetched_at_ms=stale_ms)
        req = make_request()
        results = score_routes([quote], req, now_ms=NOW_MS)
        assert results[0].is_viable is False
        assert "stale" in results[0].disqualification_reason.lower()

    def test_excess_slippage_is_disqualified(self):
        quote = make_quote(slippage_bps=500)  # 5%
        req = make_request(max_slippage_bps=100)
        results = score_routes([quote], req, now_ms=NOW_MS)
        assert results[0].is_viable is False

    def test_excess_bridge_time_is_disqualified(self):
        quote = make_quote(bridge_time_seconds=3_600)  # 1 hour
        req = make_request(max_bridge_time_seconds=900)
        results = score_routes([quote], req, now_ms=NOW_MS)
        assert results[0].is_viable is False

    def test_thin_pool_is_disqualified(self):
        # pool < 5× amount_in_usd → disqualified
        quote = make_quote(pool_liquidity_usd=1_000.0)
        req = make_request(amount_in_usd=1_000.0)  # floor = 5000
        results = score_routes([quote], req, now_ms=NOW_MS)
        assert results[0].is_viable is False

    def test_viable_routes_sorted_before_disqualified(self):
        stale = make_quote(route_id="stale", fetched_at_ms=0)
        good = make_quote(route_id="good")
        req = make_request()
        results = score_routes([stale, good], req, now_ms=NOW_MS)
        assert results[0].quote.route_id == "good"
        assert results[1].quote.route_id == "stale"

    def test_higher_profit_route_ranks_first(self):
        # Route A: more output
        route_a = make_quote(route_id="A", amount_out=18_500.0, amount_out_min=18_315.0,
                              slippage_bps=50, gas_usd=0.20)
        # Route B: less output
        route_b = make_quote(route_id="B", amount_out=18_300.0, amount_out_min=18_117.0,
                              slippage_bps=50, gas_usd=0.20)
        req = make_request()
        results = score_routes([route_b, route_a], req, now_ms=NOW_MS)
        assert results[0].quote.route_id == "A"

    def test_select_optimal_returns_none_when_no_viable_routes(self):
        stale = make_quote(fetched_at_ms=0)
        req = make_request()
        results = score_routes([stale], req, now_ms=NOW_MS)
        optimal = select_optimal_route(results)
        assert optimal is None

    def test_select_optimal_returns_best_viable(self):
        route_a = make_quote(route_id="A", amount_out=18_500.0, amount_out_min=18_315.0)
        route_b = make_quote(route_id="B", amount_out=18_300.0, amount_out_min=18_117.0)
        req = make_request()
        results = score_routes([route_a, route_b], req, now_ms=NOW_MS)
        optimal = select_optimal_route(results)
        assert optimal is not None
        assert optimal.quote.route_id == "A"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

class TestQuoteRequestValidation:
    def test_same_token_in_out_rejected(self):
        with pytest.raises(Exception):
            QuoteRequest(
                token_in="ETH", token_out="ETH",
                amount_in=1.0, amount_in_usd=3000.0,
                target_price_usd=1.0, chains=[Chain.ETHEREUM],
            )

    def test_duplicate_chains_rejected(self):
        with pytest.raises(Exception):
            QuoteRequest(
                token_in="ETH", token_out="USDC",
                amount_in=1.0, amount_in_usd=3000.0,
                target_price_usd=1.0,
                chains=[Chain.ETHEREUM, Chain.ETHEREUM],
            )

    def test_zero_amount_rejected(self):
        with pytest.raises(Exception):
            QuoteRequest(
                token_in="ETH", token_out="USDC",
                amount_in=0.0, amount_in_usd=0.0,
                target_price_usd=1.0, chains=[Chain.ETHEREUM],
            )
