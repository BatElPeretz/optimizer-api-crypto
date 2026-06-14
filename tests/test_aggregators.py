"""
tests/test_aggregators.py
-------------------------
Unit tests for the aggregator layer.

All network I/O is replaced with httpx mock transports — zero real HTTP
calls, deterministic responses, and fast execution.

Test philosophy:
  - Each test exercises one logical concern: parsing, error handling,
    retry behaviour, or field mapping.
  - The raw JSON fixtures mirror the exact structure of real API responses
    so bugs in parsing logic surface here before they surface in production.
  - We do NOT test that the APIs are up — that belongs in integration tests
    run against a staging environment, not in the unit suite.

Run with:  pytest tests/test_aggregators.py -v
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.aggregator.constants import (
    CHAIN_ID,
    raw_to_float,
    token_address,
    token_decimals,
)
from app.aggregator.gas import (
    _parse_fee_history,
    _fetch_eth_price,
    fetch_gas_snapshots,
)
from app.aggregator.oneinch import (
    _parse_response as oneinch_parse,
    _derive_slippage_bps,
    _estimate_gas_usd_from_units,
    fetch_quotes as oneinch_fetch,
)
from app.aggregator.lifi import (
    _parse_response as lifi_parse,
    _classify_route_type,
    _extract_gas_usd,
    _extract_bridge_fee_usd,
    _route_pairs,
    fetch_quotes as lifi_fetch,
)
from app.optimizer.models import (
    AggregatorSource,
    Chain,
    GasSnapshot,
    QuoteRequest,
    RouteQuote,
    RouteType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(
    token_in: str = "ETH",
    token_out: str = "USDC",
    amount_in: float = 5.0,
    amount_in_usd: float = 18_500.0,
    target_price_usd: float = 1.0,
    chains: Optional[list[Chain]] = None,
) -> QuoteRequest:
    return QuoteRequest(
        token_in=token_in,
        token_out=token_out,
        amount_in=amount_in,
        amount_in_usd=amount_in_usd,
        target_price_usd=target_price_usd,
        chains=chains or [Chain.ARBITRUM],
    )


def mock_transport(responses: list[tuple[int, dict | str]]) -> httpx.MockTransport:
    """
    Build an httpx.MockTransport that returns the given (status, body) pairs
    in order.  Works for both GET and POST requests.
    """
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        status, body = responses[idx]
        if isinstance(body, dict):
            content = json.dumps(body).encode()
            headers = {"Content-Type": "application/json"}
        else:
            content = body.encode()
            headers = {"Content-Type": "application/json"}
        return httpx.Response(status_code=status, content=content, headers=headers)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# constants.py tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_token_address_known_pair(self):
        addr = token_address("ETH", Chain.ARBITRUM)
        assert addr == "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEEE"

    def test_token_address_usdc_per_chain(self):
        eth_addr = token_address("USDC", Chain.ETHEREUM)
        arb_addr = token_address("USDC", Chain.ARBITRUM)
        # Different canonical deployments per chain
        assert eth_addr != arb_addr

    def test_token_address_case_insensitive(self):
        assert token_address("eth", Chain.ARBITRUM) == token_address("ETH", Chain.ARBITRUM)

    def test_token_address_unknown_token_raises(self):
        with pytest.raises(KeyError, match="not in the address table"):
            token_address("UNKNOWN_TOKEN_XYZ", Chain.ETHEREUM)

    def test_token_address_unknown_chain_raises(self):
        with pytest.raises(KeyError, match="no address entry for chain"):
            # USDC on Polygon is in the table, but let's use BASE for USDT
            # which is intentionally missing
            token_address("USDT", Chain.BASE)

    def test_raw_to_float_usdc(self):
        # 5 USDC = 5_000_000 raw (6 decimals)
        assert raw_to_float(5_000_000, "USDC") == 5.0

    def test_raw_to_float_eth(self):
        # 5 ETH = 5 * 10^18 raw
        raw = 5 * (10 ** 18)
        assert raw_to_float(raw, "ETH") == 5.0

    def test_raw_to_float_string_input(self):
        # APIs often return raw amounts as strings
        assert raw_to_float("1000000", "USDC") == 1.0

    def test_token_decimals_known(self):
        assert token_decimals("USDC") == 6
        assert token_decimals("ETH")  == 18

    def test_token_decimals_unknown_defaults_18(self):
        assert token_decimals("SOME_NEW_TOKEN") == 18


# ---------------------------------------------------------------------------
# gas.py — pure parsing tests (no network)
# ---------------------------------------------------------------------------

class TestParseFeeHistory:
    """Tests for the _parse_fee_history() pure function — no mocking needed."""

    def _make_fee_history(
        self,
        base_fees_hex: list[str],
        rewards_hex: list[list[str]],
    ) -> dict:
        return {
            "baseFeePerGas": base_fees_hex,
            "reward": rewards_hex,
        }

    def test_parses_base_fee_from_pending_block(self):
        # eth_feeHistory returns 5 blocks + 1 pending.
        # The LAST entry (index -1) is the pending block's base fee.
        # hex(30 gwei) = hex(30_000_000_000) = "0x6fc23ac00"
        result = self._make_fee_history(
            base_fees_hex=["0x3B9ACA00", "0x3B9ACA00", "0x6FC23AC00"],
            rewards_hex=[["0x3B9ACA00"], ["0x3B9ACA00"]],
        )
        base_fee, _ = _parse_fee_history(result)
        expected_gwei = int("0x6FC23AC00", 16) / 1e9
        assert abs(base_fee - expected_gwei) < 0.001

    def test_averages_priority_fee_across_blocks(self):
        # Two blocks: 1 gwei and 3 gwei tip → average = 2 gwei
        one_gwei_hex   = hex(1_000_000_000)   # "0x3b9aca00"
        three_gwei_hex = hex(3_000_000_000)   # "0xb2d05e00"
        result = self._make_fee_history(
            base_fees_hex=["0x3B9ACA00", "0x3B9ACA00", "0x3B9ACA00"],
            rewards_hex=[[one_gwei_hex], [three_gwei_hex]],
        )
        _, priority_fee = _parse_fee_history(result)
        assert abs(priority_fee - 2.0) < 0.001

    def test_empty_base_fees_returns_zeros(self):
        result = {"baseFeePerGas": [], "reward": []}
        base_fee, priority_fee = _parse_fee_history(result)
        assert base_fee == 0.0
        assert priority_fee == 0.0

    def test_empty_rewards_uses_fallback_priority_fee(self):
        # No reward data → fallback = 1.0 gwei
        result = {"baseFeePerGas": [hex(30_000_000_000)], "reward": []}
        _, priority_fee = _parse_fee_history(result)
        assert priority_fee == 1.0


# ---------------------------------------------------------------------------
# gas.py — fetch_gas_snapshots() with mocked HTTP
# ---------------------------------------------------------------------------

class TestFetchGasSnapshots:
    # Reusable fake fee history RPC response (Arbitrum chain)
    FAKE_FEE_HISTORY = {
        "jsonrpc": "2.0",
        "id": CHAIN_ID[Chain.ARBITRUM],
        "result": {
            "baseFeePerGas": [
                hex(100_000_000),    # 0.1 gwei
                hex(100_000_000),
                hex(100_000_000),
            ],
            "reward": [[hex(10_000_000)], [hex(10_000_000)]],   # 0.01 gwei tip
        },
    }

    FAKE_COINGECKO = {"ethereum": {"usd": 3000.0}}

    @pytest.mark.asyncio
    async def test_returns_snapshot_for_successful_chain(self):
        responses = [
            (200, self.FAKE_COINGECKO),       # CoinGecko ETH price
            (200, self.FAKE_FEE_HISTORY),     # Arbitrum RPC
        ]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        result = await fetch_gas_snapshots([Chain.ARBITRUM], client=client)

        assert Chain.ARBITRUM in result
        snap = result[Chain.ARBITRUM]
        assert isinstance(snap, GasSnapshot)
        assert snap.eth_price_usd == 3000.0
        assert snap.base_fee_gwei > 0

    @pytest.mark.asyncio
    async def test_skips_chain_when_rpc_returns_500(self):
        responses = [
            (200, self.FAKE_COINGECKO),
            (500, {"error": "internal server error"}),  # RPC failure
        ]
        # Need enough responses for retry attempts
        all_responses = [
            (200, self.FAKE_COINGECKO),
            (500, {"error": "internal"}),
            (500, {"error": "internal"}),
            (500, {"error": "internal"}),
        ]
        client = httpx.AsyncClient(transport=mock_transport(all_responses))
        result = await fetch_gas_snapshots([Chain.ARBITRUM], client=client)
        # Chain should be absent — not crashed, just skipped
        assert Chain.ARBITRUM not in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_eth_price_is_zero(self):
        bad_price_response = {"ethereum": {"usd": 0.0}}
        responses = [(200, bad_price_response)]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        result = await fetch_gas_snapshots([Chain.ARBITRUM], client=client)
        assert result == {}

    @pytest.mark.asyncio
    async def test_fetches_multiple_chains_in_parallel(self):
        # Supply enough responses for: 1 CoinGecko + 2 RPC calls
        responses = [
            (200, self.FAKE_COINGECKO),
            (200, self.FAKE_FEE_HISTORY),
            (200, self.FAKE_FEE_HISTORY),
        ]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        result = await fetch_gas_snapshots(
            [Chain.ARBITRUM, Chain.BASE], client=client
        )
        # Both chains should have snapshots (transport returns same data for both)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# oneinch.py — pure helpers
# ---------------------------------------------------------------------------

class TestOneInchHelpers:
    def test_derive_slippage_zero_impact(self):
        # If quoted output equals expected output → 0 slippage from signal A
        bps = _derive_slippage_bps(
            amount_in_usd=1000.0,
            amount_out_usd=1000.0,
            expected_out_usd=1000.0,
            pool_liquidity_usd=50_000_000.0,  # deep pool → low signal B
        )
        assert bps >= 0

    def test_derive_slippage_large_price_impact(self):
        # Quoted output is 5% below expected → signal A = 500 bps
        bps = _derive_slippage_bps(
            amount_in_usd=10_000.0,
            amount_out_usd=9_500.0,
            expected_out_usd=10_000.0,
            pool_liquidity_usd=100_000_000.0,
        )
        assert bps >= 500

    def test_estimate_gas_usd_arbitrum_cheaper_than_ethereum(self):
        arb_cost = _estimate_gas_usd_from_units(Chain.ARBITRUM, 150_000)
        eth_cost = _estimate_gas_usd_from_units(Chain.ETHEREUM, 150_000)
        assert arb_cost < eth_cost / 10

    def test_estimate_gas_usd_scales_with_units(self):
        cost_100k = _estimate_gas_usd_from_units(Chain.ETHEREUM, 100_000)
        cost_200k = _estimate_gas_usd_from_units(Chain.ETHEREUM, 200_000)
        assert abs(cost_200k - 2 * cost_100k) < 1e-10


# ---------------------------------------------------------------------------
# oneinch.py — _parse_response()
# ---------------------------------------------------------------------------

class TestOneInchParseResponse:
    # Minimal realistic 1inch v5.2 quote response
    FAKE_1INCH = {
        "toAmount": "18_400_123_456".replace("_", ""),   # 18400.123456 USDC (6 dec)
        "gas":      180_000,
    }

    def _req(self) -> QuoteRequest:
        return make_request()

    def test_returns_route_quote(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert isinstance(result, RouteQuote)

    def test_amount_out_converted_from_raw(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        # 18400123456 / 10^6 = 18400.123456
        assert abs(result.amount_out - 18400.123456) < 0.001

    def test_amount_out_min_is_below_amount_out(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.amount_out_min < result.amount_out

    def test_source_is_oneinch(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.source == AggregatorSource.ONEINCH

    def test_route_type_is_direct_swap(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.route_type == RouteType.DIRECT_SWAP

    def test_chain_in_equals_chain_out(self):
        # 1inch is always single-chain
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.chain_in == result.chain_out == Chain.ARBITRUM

    def test_bridge_fields_are_zero(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.bridge_fee_usd == 0.0
        assert result.bridge_time_seconds == 0

    def test_gas_usd_is_positive(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.gas_usd > 0

    def test_slippage_bps_is_non_negative(self):
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        assert result.slippage_bps >= 0

    def test_returns_none_on_missing_key(self):
        result = oneinch_parse({"unexpected_field": "value"}, Chain.ARBITRUM, self._req())
        assert result is None

    def test_returns_none_on_invalid_amount(self):
        result = oneinch_parse({"toAmount": "not_a_number", "gas": 0}, Chain.ARBITRUM, self._req())
        assert result is None

    def test_fetched_at_ms_is_recent(self):
        before = int(time.time() * 1000)
        result = oneinch_parse(self.FAKE_1INCH, Chain.ARBITRUM, self._req())
        after  = int(time.time() * 1000)
        assert result is not None
        assert before <= result.fetched_at_ms <= after


# ---------------------------------------------------------------------------
# oneinch.py — fetch_quotes() with mocked HTTP
# ---------------------------------------------------------------------------

class TestOneInchFetch:
    FAKE_1INCH_RESPONSE = {
        "toAmount": "18400000000",
        "gas": 180_000,
    }

    @pytest.mark.asyncio
    async def test_returns_quotes_on_success(self):
        responses = [(200, self.FAKE_1INCH_RESPONSE)]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await oneinch_fetch(req, client=client)
        assert len(quotes) == 1
        assert quotes[0].source == AggregatorSource.ONEINCH

    @pytest.mark.asyncio
    async def test_returns_empty_on_400(self):
        responses = [(400, {"description": "insufficient liquidity"})]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await oneinch_fetch(req, client=client)
        assert quotes == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_500_after_retries(self):
        # MAX_RETRIES=2 → 3 attempts total
        responses = [
            (500, {"error": "server error"}),
            (500, {"error": "server error"}),
            (500, {"error": "server error"}),
        ]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await oneinch_fetch(req, client=client)
        assert quotes == []

    @pytest.mark.asyncio
    async def test_fetches_multiple_chains_independently(self):
        # Two successful responses for two chains
        responses = [
            (200, self.FAKE_1INCH_RESPONSE),
            (200, self.FAKE_1INCH_RESPONSE),
        ]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM, Chain.BASE])
        quotes = await oneinch_fetch(req, client=client)
        assert len(quotes) == 2

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_quotes(self):
        # asyncio.gather fires both chain requests in parallel, so we can't
        # predict which response each chain gets from a sequential mock.
        # Use a URL-aware transport: ARBITRUM URL always 500s, BASE always 200s.
        arb_id  = str(CHAIN_ID[Chain.ARBITRUM])
        base_id = str(CHAIN_ID[Chain.BASE])
        success_body = json.dumps(self.FAKE_1INCH_RESPONSE).encode()
        failure_body = json.dumps({"error": "chain down"}).encode()

        def url_router(request: httpx.Request) -> httpx.Response:
            if f"/{arb_id}/" in str(request.url):
                return httpx.Response(500, content=failure_body,
                                      headers={"Content-Type": "application/json"})
            return httpx.Response(200, content=success_body,
                                  headers={"Content-Type": "application/json"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(url_router))
        req = make_request(chains=[Chain.ARBITRUM, Chain.BASE])
        quotes = await oneinch_fetch(req, client=client)
        # ARBITRUM exhausts all retries and fails; BASE succeeds
        assert len(quotes) == 1
        assert quotes[0].chain_in == Chain.BASE


# ---------------------------------------------------------------------------
# lifi.py — pure helpers
# ---------------------------------------------------------------------------

class TestLiFiHelpers:
    def test_classify_direct_swap_single_step(self):
        data = {"includedSteps": [{"type": "swap"}]}
        assert _classify_route_type(data) == RouteType.DIRECT_SWAP

    def test_classify_multi_hop_multiple_swap_steps(self):
        data = {"includedSteps": [{"type": "swap"}, {"type": "swap"}]}
        assert _classify_route_type(data) == RouteType.MULTI_HOP

    def test_classify_bridge_then_swap(self):
        data = {"includedSteps": [{"type": "cross"}, {"type": "swap"}]}
        assert _classify_route_type(data) == RouteType.BRIDGE_THEN_SWAP

    def test_classify_bridge_only(self):
        data = {"includedSteps": [{"type": "cross"}]}
        assert _classify_route_type(data) == RouteType.BRIDGE_THEN_SWAP

    def test_classify_empty_steps_defaults_direct(self):
        data = {}
        assert _classify_route_type(data) == RouteType.DIRECT_SWAP

    def test_extract_gas_usd_sums_all_steps(self):
        estimate = {"gasCosts": [{"amountUSD": "1.50"}, {"amountUSD": "0.30"}]}
        assert abs(_extract_gas_usd(estimate) - 1.80) < 0.001

    def test_extract_gas_usd_empty_returns_zero(self):
        assert _extract_gas_usd({}) == 0.0

    def test_extract_gas_usd_ignores_bad_values(self):
        estimate = {"gasCosts": [{"amountUSD": "bad"}, {"amountUSD": "1.00"}]}
        assert abs(_extract_gas_usd(estimate) - 1.00) < 0.001

    def test_extract_bridge_fee_sums_correctly(self):
        estimate = {"feeCosts": [{"amountUSD": "3.00"}, {"amountUSD": "0.21"}]}
        assert abs(_extract_bridge_fee_usd(estimate) - 3.21) < 0.001

    def test_route_pairs_includes_same_chain(self):
        pairs = _route_pairs([Chain.ARBITRUM, Chain.BASE])
        assert (Chain.ARBITRUM, Chain.ARBITRUM) in pairs
        assert (Chain.BASE, Chain.BASE) in pairs

    def test_route_pairs_includes_cross_chain(self):
        pairs = _route_pairs([Chain.ARBITRUM, Chain.BASE])
        assert (Chain.ARBITRUM, Chain.BASE) in pairs
        assert (Chain.BASE, Chain.ARBITRUM) in pairs

    def test_route_pairs_count_is_n_squared(self):
        chains = [Chain.ARBITRUM, Chain.BASE, Chain.ETHEREUM]
        pairs = _route_pairs(chains)
        assert len(pairs) == 9   # 3² = 9


# ---------------------------------------------------------------------------
# lifi.py — _parse_response()
# ---------------------------------------------------------------------------

class TestLiFiParseResponse:
    # Realistic Li.Fi response for a cross-chain ETH→USDC route
    FAKE_LIFI = {
        "estimate": {
            "toAmount":          "18400000000",   # 18400 USDC (6 dec)
            "toAmountMin":       "18032000000",   # 18032 USDC → 2% slippage
            "gasCosts": [
                {"amountUSD": "0.45"},
                {"amountUSD": "0.12"},
            ],
            "feeCosts": [
                {"amountUSD": "3.21"},
            ],
            "executionDuration": 120,             # 2 minutes bridge time
        },
        "includedSteps": [
            {"type": "cross"},
            {"type": "swap"},
        ],
    }

    def _req(self) -> QuoteRequest:
        return make_request(chains=[Chain.ETHEREUM, Chain.ARBITRUM])

    def test_returns_route_quote(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert isinstance(result, RouteQuote)

    def test_amount_out_converted_from_raw(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert abs(result.amount_out - 18400.0) < 0.001

    def test_amount_out_min_converted_from_raw(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert abs(result.amount_out_min - 18032.0) < 0.001

    def test_slippage_bps_computed_from_toamountmin(self):
        # (18400 - 18032) / 18400 × 10000 = 200 bps
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert abs(result.slippage_bps - 200) <= 1   # ±1 from int rounding

    def test_gas_usd_is_sum_of_gas_costs(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert abs(result.gas_usd - 0.57) < 0.001

    def test_bridge_fee_usd_is_sum_of_fee_costs(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert abs(result.bridge_fee_usd - 3.21) < 0.001

    def test_bridge_time_seconds_from_execution_duration(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result.bridge_time_seconds == 120

    def test_route_type_bridge_then_swap(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result.route_type == RouteType.BRIDGE_THEN_SWAP

    def test_chain_in_and_chain_out_preserved(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result.chain_in  == Chain.ETHEREUM
        assert result.chain_out == Chain.ARBITRUM

    def test_source_is_lifi(self):
        result = lifi_parse(self.FAKE_LIFI, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result.source == AggregatorSource.LIFI

    def test_returns_none_on_missing_estimate_key(self):
        result = lifi_parse({"action": {}}, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result is None

    def test_returns_none_on_invalid_to_amount(self):
        bad = {**self.FAKE_LIFI, "estimate": {**self.FAKE_LIFI["estimate"], "toAmount": "NaN"}}
        result = lifi_parse(bad, Chain.ETHEREUM, Chain.ARBITRUM, self._req())
        assert result is None

    def test_same_chain_route_has_zero_bridge_time(self):
        same_chain_data = {
            "estimate": {
                "toAmount":    "18400000000",
                "toAmountMin": "18216000000",
                "gasCosts":    [{"amountUSD": "0.30"}],
                "feeCosts":    [],
                "executionDuration": 0,
            },
            "includedSteps": [{"type": "swap"}],
        }
        result = lifi_parse(same_chain_data, Chain.ARBITRUM, Chain.ARBITRUM, self._req())
        assert result is not None
        assert result.bridge_time_seconds == 0
        assert result.bridge_fee_usd == 0.0


# ---------------------------------------------------------------------------
# lifi.py — fetch_quotes() with mocked HTTP
# ---------------------------------------------------------------------------

class TestLiFiFetch:
    FAKE_RESPONSE = {
        "estimate": {
            "toAmount":          "18400000000",
            "toAmountMin":       "18032000000",
            "gasCosts":          [{"amountUSD": "0.45"}],
            "feeCosts":          [{"amountUSD": "3.21"}],
            "executionDuration": 120,
        },
        "includedSteps": [{"type": "cross"}, {"type": "swap"}],
    }

    @pytest.mark.asyncio
    async def test_returns_quotes_on_success(self):
        # Single chain → 1 pair (ARB→ARB) → 1 response
        responses = [(200, self.FAKE_RESPONSE)]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await lifi_fetch(req, client=client)
        assert len(quotes) == 1

    @pytest.mark.asyncio
    async def test_two_chains_fire_four_requests(self):
        # 2 chains → 4 pairs (ARB→ARB, ARB→ETH, ETH→ARB, ETH→ETH)
        responses = [(200, self.FAKE_RESPONSE)] * 4
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM, Chain.ETHEREUM])
        quotes = await lifi_fetch(req, client=client)
        assert len(quotes) == 4

    @pytest.mark.asyncio
    async def test_404_skipped_gracefully(self):
        responses = [(404, {"message": "No route found"})]
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await lifi_fetch(req, client=client)
        assert quotes == []

    @pytest.mark.asyncio
    async def test_partial_failure_returns_successful_quotes(self):
        # 2 chains → 4 pairs fired in parallel.
        # Make any request that goes ARB→ETH always 500, all others 200.
        arb_id = str(CHAIN_ID[Chain.ARBITRUM])
        eth_id = str(CHAIN_ID[Chain.ETHEREUM])
        success_body = json.dumps(self.FAKE_RESPONSE).encode()
        failure_body = json.dumps({"error": "bridge down"}).encode()

        def url_router(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            params = dict(request.url.params)
            from_chain = params.get("fromChain", "")
            to_chain   = params.get("toChain", "")
            # Fail the ARB→ETH pair only
            if str(from_chain) == arb_id and str(to_chain) == eth_id:
                return httpx.Response(500, content=failure_body,
                                      headers={"Content-Type": "application/json"})
            return httpx.Response(200, content=success_body,
                                  headers={"Content-Type": "application/json"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(url_router))
        req = make_request(chains=[Chain.ARBITRUM, Chain.ETHEREUM])
        quotes = await lifi_fetch(req, client=client)
        # 4 pairs - 1 failing (after retries) = 3 successful quotes
        assert len(quotes) == 3

    @pytest.mark.asyncio
    async def test_all_failures_returns_empty_list(self):
        responses = [(500, {"error": "down"})] * 12  # enough for all retries
        client = httpx.AsyncClient(transport=mock_transport(responses))
        req = make_request(chains=[Chain.ARBITRUM])
        quotes = await lifi_fetch(req, client=client)
        assert quotes == []
