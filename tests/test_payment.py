"""
tests/test_payment.py
---------------------
Unit tests for the L402 payment layer.

All tests are fully offline — Coinos and Lightning Network are mocked.
The macaroon crypto tests use known inputs so you can verify the math
independently.

Run with:  pytest tests/test_payment.py -v
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.payment.coinos import CoinosError, LightningInvoice, create_invoice, PRICE_SATS
from app.payment.l402 import (
    MACAROON_TTL_SECONDS,
    _decode_macaroon,
    _parse_l402_header,
    _sign,
    l402_gate,
    mint_macaroon,
    verify_macaroon,
)


# ---------------------------------------------------------------------------
# Environment setup for tests
# ---------------------------------------------------------------------------

TEST_SECRET = "a" * 32   # deterministic 32-char secret for tests
TEST_PAYMENT_HASH = "abc123def456" * 4 + "aabb"   # 50-char fake hash

# Patch L402_SECRET into the environment for the entire test module.
# Each test class that calls crypto functions must set this.
os.environ.setdefault("L402_SECRET", TEST_SECRET)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_valid_preimage_and_hash() -> tuple[str, str]:
    """
    Return a (preimage_hex, payment_hash) pair where SHA256(preimage)==hash.
    Used for tests that need a cryptographically valid payment proof.
    """
    preimage_bytes = b"test_preimage_for_unit_tests_only"
    payment_hash   = hashlib.sha256(preimage_bytes).hexdigest()
    preimage_hex   = preimage_bytes.hex()
    return preimage_hex, payment_hash


def make_valid_macaroon(payment_hash: str, offset_seconds: int = 0) -> str:
    """
    Mint a macaroon with the test secret, optionally shifted in time.
    offset_seconds < 0 → already expired.
    """
    secret  = TEST_SECRET.encode()
    expiry  = int(time.time()) + MACAROON_TTL_SECONDS + offset_seconds
    sig     = _sign(payment_hash, expiry, secret)
    raw     = f"{payment_hash}:{expiry}:{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


# ---------------------------------------------------------------------------
# coinos.py tests
# ---------------------------------------------------------------------------

class TestCoinosCreateInvoice:
    FAKE_RESPONSE = {
        "hash":    "deadbeef" * 8,
        "text":    "lnbc100n1ptest...",
        "id":      "uuid-1234",
        "amount":  PRICE_SATS,
    }

    @pytest.mark.asyncio
    async def test_returns_lightning_invoice_on_success(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self.FAKE_RESPONSE)

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            result = await create_invoice(client=client)

        assert isinstance(result, LightningInvoice)
        assert result.payment_hash == self.FAKE_RESPONSE["hash"]
        assert result.bolt11       == self.FAKE_RESPONSE["text"]
        assert result.amount_sats  == PRICE_SATS

    @pytest.mark.asyncio
    async def test_raises_coinos_error_on_401(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        with patch.dict(os.environ, {"COINOS_API_KEY": "bad-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with pytest.raises(CoinosError, match="401"):
                await create_invoice(client=client)

    @pytest.mark.asyncio
    async def test_raises_coinos_error_on_missing_fields(self):
        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "ok"})  # missing hash & text

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with pytest.raises(CoinosError, match="missing 'hash' or 'text'"):
                await create_invoice(client=client)

    @pytest.mark.asyncio
    async def test_raises_coinos_error_when_api_key_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "COINOS_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(CoinosError, match="COINOS_API_KEY"):
                await create_invoice()

    @pytest.mark.asyncio
    async def test_raises_coinos_error_on_timeout(self):
        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timeout", request=req)

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with pytest.raises(CoinosError, match="timed out"):
                await create_invoice(client=client)

    @pytest.mark.asyncio
    async def test_custom_amount_is_sent_in_request_body(self):
        captured = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json=self.FAKE_RESPONSE)

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            await create_invoice(amount_sats=42, client=client)

        # Payload must be wrapped: {"invoice": {"amount": ..., "type": "lightning"}}
        assert "invoice" in captured["body"], "outer 'invoice' wrapper missing from payload"
        assert captured["body"]["invoice"]["amount"] == 42

    @pytest.mark.asyncio
    async def test_payload_has_outer_invoice_wrapper(self):
        """Regression test for Coinos 500: 'invoice.own = !0' on undefined."""
        captured = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(req.content)
            return httpx.Response(200, json=self.FAKE_RESPONSE)

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            await create_invoice(client=client)

        body = captured["body"]
        # Top-level must be {"invoice": {...}}, not {"amount": ..., "type": ...}
        assert "invoice" in body
        assert "amount" not in body
        assert body["invoice"]["type"] == "lightning"

    @pytest.mark.asyncio
    async def test_nested_invoice_response_is_handled(self):
        """Some Coinos API versions return {"invoice": {"hash": ..., "text": ...}}."""
        nested_response = {"invoice": self.FAKE_RESPONSE}

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=nested_response)

        with patch.dict(os.environ, {"COINOS_API_KEY": "test-key"}):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            result = await create_invoice(client=client)

        assert result.payment_hash == self.FAKE_RESPONSE["hash"]
        assert result.bolt11       == self.FAKE_RESPONSE["text"]


# ---------------------------------------------------------------------------
# l402.py — macaroon crypto
# ---------------------------------------------------------------------------

class TestMintMacaroon:
    def test_returns_base64url_string(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon("abc123")
        # base64url uses - and _ not + and /
        assert "+" not in token
        assert "/" not in token

    def test_decodes_to_three_parts(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon("abc123")
        payment_hash, expiry, sig = _decode_macaroon(token)
        assert payment_hash == "abc123"
        assert sig != ""

    def test_expiry_is_in_the_future(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon("abc123")
        _, expiry, _ = _decode_macaroon(token)
        assert expiry > int(time.time())

    def test_expiry_respects_ttl(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            before = int(time.time())
            token  = mint_macaroon("abc123")
            after  = int(time.time())
        _, expiry, _ = _decode_macaroon(token)
        assert before + MACAROON_TTL_SECONDS <= expiry <= after + MACAROON_TTL_SECONDS + 1

    def test_different_hashes_produce_different_tokens(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            t1 = mint_macaroon("hash_one")
            t2 = mint_macaroon("hash_two")
        assert t1 != t2

    def test_raises_on_short_secret(self):
        with patch.dict(os.environ, {"L402_SECRET": "short"}):
            with pytest.raises(RuntimeError, match="L402_SECRET"):
                mint_macaroon("abc")


class TestVerifyMacaroon:
    def test_valid_macaroon_and_preimage_passes(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
            valid, reason = verify_macaroon(token, preimage_hex)
        assert valid is True
        assert reason == ""

    def test_wrong_preimage_fails(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        wrong_preimage = "deadbeef" * 8   # doesn't hash to payment_hash
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
            valid, reason = verify_macaroon(token, wrong_preimage)
        assert valid is False
        assert "Preimage does not match" in reason

    def test_expired_macaroon_fails(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            # Mint a macaroon that expired 1 second ago
            token = make_valid_macaroon(payment_hash, offset_seconds=-MACAROON_TTL_SECONDS - 1)
            valid, reason = verify_macaroon(token, preimage_hex)
        assert valid is False
        assert "expired" in reason.lower()

    def test_tampered_signature_fails(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
        # Flip the last character of the token to tamper with the signature
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            valid, reason = verify_macaroon(tampered, preimage_hex)
        assert valid is False

    def test_malformed_token_fails(self):
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            valid, reason = verify_macaroon("not_valid_base64!!!", "aabb")
        assert valid is False
        assert "Malformed" in reason

    def test_non_hex_preimage_fails(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
            valid, reason = verify_macaroon(token, "not-hex-!!!")
        assert valid is False
        assert "not valid hex" in reason


class TestParseL402Header:
    def test_valid_header_returns_tuple(self):
        result = _parse_l402_header("L402 mymacaroon:preimage64chars")
        assert result == ("mymacaroon", "preimage64chars")

    def test_case_insensitive_scheme(self):
        result = _parse_l402_header("l402 mac:pre")
        assert result == ("mac", "pre")

    def test_empty_header_returns_none(self):
        assert _parse_l402_header("") is None

    def test_wrong_scheme_returns_none(self):
        assert _parse_l402_header("Bearer sometoken") is None

    def test_missing_colon_returns_none(self):
        assert _parse_l402_header("L402 nocolon") is None

    def test_splits_on_last_colon(self):
        # Macaroon base64 may contain '=' padding but not ':', so
        # the split should give the entire macaroon before the last colon
        result = _parse_l402_header("L402 abc:def:preimage")
        assert result is not None
        macaroon, preimage = result
        assert macaroon  == "abc:def"
        assert preimage  == "preimage"


# ---------------------------------------------------------------------------
# l402.py — FastAPI integration (TestClient = synchronous wrapper)
# ---------------------------------------------------------------------------

def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app with the l402_gate protecting one route."""
    from fastapi import Depends
    test_app = FastAPI()

    @test_app.get("/protected")
    async def protected(_: None = Depends(l402_gate)):
        return {"access": "granted"}

    return test_app


# Shared app instance — built once, reused across all TestL402Gate tests.
_TEST_APP = _build_test_app()

# httpx.AsyncClient with ASGITransport is the correct way to test ASGI apps
# against httpx >= 0.23.  Starlette's sync TestClient is incompatible with
# httpx >= 0.28 (which removed the 'app=' kwarg from httpx.Client.__init__).
async def _asgi_get(path: str, headers: Optional[dict] = None) -> httpx.Response:
    """Fire a GET against the test ASGI app and return the response."""
    transport = httpx.AsyncHTTPTransport()   # placeholder — overridden below
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_TEST_APP),
        base_url="http://testserver",
    ) as client:
        return await client.get(path, headers=headers or {})


class TestL402Gate:
    FAKE_INVOICE = LightningInvoice(
        payment_hash=make_valid_preimage_and_hash()[1],
        bolt11="lnbc100n1ptest...",
        amount_sats=PRICE_SATS,
    )

    @pytest.mark.asyncio
    async def test_no_auth_header_returns_402(self):
        with patch("app.payment.l402.create_invoice", new_callable=AsyncMock) as mock_invoice, \
             patch.dict(os.environ, {"L402_SECRET": TEST_SECRET, "COINOS_API_KEY": "key"}):
            mock_invoice.return_value = self.FAKE_INVOICE
            resp = await _asgi_get("/protected")
        assert resp.status_code == 402

    @pytest.mark.asyncio
    async def test_402_response_includes_www_authenticate_header(self):
        with patch("app.payment.l402.create_invoice", new_callable=AsyncMock) as mock_invoice, \
             patch.dict(os.environ, {"L402_SECRET": TEST_SECRET, "COINOS_API_KEY": "key"}):
            mock_invoice.return_value = self.FAKE_INVOICE
            resp = await _asgi_get("/protected")
        assert "www-authenticate" in resp.headers
        assert resp.headers["www-authenticate"].startswith("L402 macaroon=")

    @pytest.mark.asyncio
    async def test_402_body_contains_bolt11_invoice(self):
        with patch("app.payment.l402.create_invoice", new_callable=AsyncMock) as mock_invoice, \
             patch.dict(os.environ, {"L402_SECRET": TEST_SECRET, "COINOS_API_KEY": "key"}):
            mock_invoice.return_value = self.FAKE_INVOICE
            resp = await _asgi_get("/protected")
        body = resp.json()
        assert "invoice" in body["detail"]
        assert body["detail"]["invoice"] == self.FAKE_INVOICE.bolt11

    @pytest.mark.asyncio
    async def test_valid_l402_header_grants_access(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
        auth = f"L402 {token}:{preimage_hex}"
        resp = await _asgi_get("/protected", headers={"Authorization": auth})
        assert resp.status_code == 200
        assert resp.json() == {"access": "granted"}

    @pytest.mark.asyncio
    async def test_wrong_preimage_returns_401(self):
        _, payment_hash = make_valid_preimage_and_hash()
        wrong_preimage  = "ff" * 32   # valid hex, wrong hash
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = mint_macaroon(payment_hash)
        auth = f"L402 {token}:{wrong_preimage}"
        resp = await _asgi_get("/protected", headers={"Authorization": auth})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_macaroon_returns_401(self):
        preimage_hex, payment_hash = make_valid_preimage_and_hash()
        with patch.dict(os.environ, {"L402_SECRET": TEST_SECRET}):
            token = make_valid_macaroon(payment_hash, offset_seconds=-MACAROON_TTL_SECONDS - 1)
        auth = f"L402 {token}:{preimage_hex}"
        resp = await _asgi_get("/protected", headers={"Authorization": auth})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_coinos_down_returns_503(self):
        with patch("app.payment.l402.create_invoice", new_callable=AsyncMock) as mock_invoice, \
             patch.dict(os.environ, {"L402_SECRET": TEST_SECRET, "COINOS_API_KEY": "key"}):
            mock_invoice.side_effect = CoinosError("Coinos is down")
            resp = await _asgi_get("/protected")
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_l402_secret_returns_500_on_challenge(self):
        env = {k: v for k, v in os.environ.items() if k != "L402_SECRET"}
        env["COINOS_API_KEY"] = "key"
        with patch("app.payment.l402.create_invoice", new_callable=AsyncMock) as mock_invoice, \
             patch.dict(os.environ, env, clear=True):
            mock_invoice.return_value = self.FAKE_INVOICE
            resp = await _asgi_get("/protected")
        assert resp.status_code == 500
