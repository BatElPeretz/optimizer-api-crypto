"""
payment/coinos.py
-----------------
Async client for the Coinos Lightning Network API (coinos.io/api).

Coinos is a free, custodial Lightning wallet with a simple REST API that
requires no node setup.  We use it exclusively to generate BOLT11 invoices
that agents pay as part of the L402 protocol.

We never hold funds here.  The flow is:
  1. Agent hits /v1/quote without payment.
  2. We call Coinos to mint a fresh invoice for PRICE_SATS satoshis.
  3. We return the invoice + our macaroon in the 402 WWW-Authenticate header.
  4. Agent pays the invoice on Lightning, then retries with the preimage.
  5. l402.py verifies the preimage — Coinos is never contacted again.

Setup:
  - Create a free account at coinos.io
  - Go to Settings → API → generate a token
  - Set COINOS_API_KEY=<token> in your .env file

Coinos invoice endpoint:
  POST https://coinos.io/api/invoice
  Headers: Authorization: Bearer <token>
           Content-Type: application/json

  Correct payload shape (the outer "invoice" wrapper is required):
  {
    "invoice": {
      "amount": <sats>,
      "type":   "lightning"
    }
  }

  Without the wrapper their Express handler crashes evaluating
  `invoice.own = true` on an undefined value (the 500 you saw).

Response — Coinos returns the invoice object directly at the top level:
  {
    "hash":    "abc123...",      ← payment_hash (SHA256 of preimage)
    "text":    "lnbc100n1...",   ← BOLT11 invoice string
    "id":      "uuid",
    "amount":  100,
    "type":    "lightning"
  }
  Some API versions nest this under an "invoice" key; we handle both.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COINOS_BASE_URL  = "https://coinos.io/api"
REQUEST_TIMEOUT  = 8.0   # seconds — invoice creation is not on the hot-path

# Price per API call in satoshis.
# 10 sats ≈ $0.006 at BTC=$60k — cheap enough for agents, covers costs.
# Raise to charge a percentage of the profit the route generates (advanced).
PRICE_SATS: int = int(os.environ.get("OPTIMIZER_PRICE_SATS", "10"))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LightningInvoice:
    """
    Everything the L402 layer needs to issue a payment challenge.

    payment_hash   — the SHA256 preimage hash embedded in the BOLT11 invoice.
                     The agent's wallet provides the preimage when settling.
                     l402.py verifies: SHA256(preimage) == payment_hash.
    bolt11         — the full BOLT11 invoice string starting with "lnbc".
                     Passed verbatim in the WWW-Authenticate header.
    amount_sats    — the amount the agent must pay (for logging/auditing).
    """
    payment_hash: str
    bolt11: str
    amount_sats: int


class CoinosError(Exception):
    """Raised when the Coinos API returns an unexpected response."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """
    Read the Coinos API key from the environment.
    Raises at call time (not import time) so the app starts even when the
    key is absent — useful during unit tests that mock this function.
    """
    key = os.environ.get("COINOS_API_KEY", "")
    if not key:
        raise CoinosError(
            "COINOS_API_KEY is not set. "
            "Create a free account at coinos.io, generate an API token "
            "in Settings → API, then add COINOS_API_KEY=<token> to your .env file."
        )
    return key


async def create_invoice(
    amount_sats: Optional[int] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> LightningInvoice:
    """
    Create a Lightning invoice via the Coinos REST API.

    Args:
        amount_sats: Satoshi amount to request.  Defaults to PRICE_SATS.
        client:      Optional shared httpx.AsyncClient.  A new one is created
                     and closed internally if not provided.

    Returns:
        LightningInvoice with the BOLT11 string and payment_hash.

    Raises:
        CoinosError: If the API key is missing, the request fails, or the
                     response is malformed.  The caller (l402.py) catches
                     this and returns a 503 instead of a 402 so the agent
                     knows the payment system is unavailable — not that it
                     owes money it can't pay.
    """
    sats = amount_sats if amount_sats is not None else PRICE_SATS
    api_key = _get_api_key()

    # The outer "invoice" wrapper is mandatory — Coinos's Express handler
    # does `invoice.own = true` where `invoice` is `req.body.invoice`.
    # Sending the fields at the top level leaves `invoice` undefined and
    # produces a 500: "undefined is not an object (evaluating 'invoice.own = !0')".
    payload = {
        "invoice": {
            "amount": sats,
            "type":   "lightning",
        }
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        resp = await client.post(
            f"{COINOS_BASE_URL}/invoice",
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )

        if resp.status_code == 401:
            raise CoinosError(
                "Coinos rejected our API key (401). "
                "Verify COINOS_API_KEY is correct in your .env file."
            )

        if resp.status_code == 429:
            raise CoinosError("Coinos rate-limit hit (429). Retry after a short delay.")

        resp.raise_for_status()
        data = resp.json()

        # Coinos returns the invoice object at the top level in current API
        # versions, but some builds nest it under an "invoice" key.
        # We check the top level first, then fall back to the nested shape.
        invoice_data = data if "hash" in data else data.get("invoice", data)

        payment_hash = invoice_data.get("hash", "")
        bolt11       = invoice_data.get("text", "")

        if not payment_hash or not bolt11:
            raise CoinosError(
                f"Coinos response missing 'hash' or 'text' fields. "
                f"Top-level keys: {list(data.keys())}. Full response: {data}"
            )

        logger.info(
            "Coinos invoice created | amount=%d sats | hash=%s...%s",
            sats, payment_hash[:8], payment_hash[-4:],
        )

        return LightningInvoice(
            payment_hash=payment_hash,
            bolt11=bolt11,
            amount_sats=sats,
        )

    except httpx.TimeoutException as exc:
        raise CoinosError(f"Coinos request timed out after {REQUEST_TIMEOUT}s: {exc}") from exc

    except httpx.ConnectError as exc:
        raise CoinosError(f"Could not connect to Coinos API: {exc}") from exc

    except httpx.HTTPStatusError as exc:
        raise CoinosError(
            f"Coinos returned HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        ) from exc

    finally:
        if own_client:
            await client.aclose()
