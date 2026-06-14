"""
payment/l402.py
---------------
L402 Lightning payment protocol — macaroon minting, verification,
and the FastAPI dependency that guards every paid endpoint.

Protocol overview (RFC-style):
  1. Agent sends request with no Authorization header.
  2. Server calls Coinos, gets a fresh BOLT11 invoice and its payment_hash.
  3. Server mints a macaroon tied to that payment_hash.
  4. Server responds 402 with:
         WWW-Authenticate: L402 macaroon="<token>", invoice="<bolt11>"
  5. Agent pays the invoice on Lightning, receives the preimage.
  6. Agent retries the request with:
         Authorization: L402 <macaroon>:<preimage_hex>
  7. Server verifies:
         a. Macaroon signature is valid (we signed it with our secret).
         b. SHA256(preimage) == payment_hash stored inside the macaroon.
         c. Macaroon has not expired.
  8. If all checks pass → proceed.  Otherwise → 401 Unauthorized.

Macaroon format (custom lightweight implementation):
  We intentionally avoid the full Macaroon spec (with caveats and
  third-party discharge) to keep the MVP dependency-free.  Our token is:

      base64url( payment_hash + ":" + expiry_unix + ":" + hmac )

  where hmac = HMAC-SHA256(L402_SECRET, payment_hash + ":" + expiry_unix)

  This is unforgeable as long as L402_SECRET stays secret, and it binds
  the macaroon to exactly one payment_hash.

Security notes:
  - L402_SECRET must be at least 32 bytes of entropy.  Generate with:
        python -c "import secrets; print(secrets.token_hex(32))"
  - Tokens expire after MACAROON_TTL_SECONDS.  Agents that take longer to
    pay than this TTL must request a fresh invoice.
  - We use HMAC-SHA256, not a simple hash, so the signature cannot be
    brute-forced even if the payment_hash is known.
  - secrets.compare_digest prevents timing-attack comparison of HMACs.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.payment.coinos import CoinosError, LightningInvoice, create_invoice

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How long (seconds) a minted macaroon is valid for payment.
# 600s = 10 minutes.  Agents are fast; this is generous.
MACAROON_TTL_SECONDS: int = int(os.environ.get("MACAROON_TTL_SECONDS", "600"))


def _get_l402_secret() -> bytes:
    """
    Read L402_SECRET from the environment and return as bytes.

    Raises at call time so tests can patch os.environ without side effects
    at import time.  In production this is called on every request — cheap
    because os.environ lookups are O(1).
    """
    secret = os.environ.get("L402_SECRET", "")
    if not secret or len(secret) < 32:
        raise RuntimeError(
            "L402_SECRET is missing or too short (need ≥32 chars). "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            " and add it to your .env file."
        )
    return secret.encode()


# ---------------------------------------------------------------------------
# Macaroon helpers
# ---------------------------------------------------------------------------

def _sign(payment_hash: str, expiry: int, secret: bytes) -> str:
    """
    Compute HMAC-SHA256 over the canonical payload.

    Payload: "{payment_hash}:{expiry}"  — both fields together prevent
    an attacker from swapping expiry values between macaroons.
    """
    payload = f"{payment_hash}:{expiry}".encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def mint_macaroon(payment_hash: str) -> str:
    """
    Create a signed, expiring macaroon tied to one Lightning payment_hash.

    Returns a base64url-encoded string safe for use in HTTP headers.
    """
    secret = _get_l402_secret()
    expiry = int(time.time()) + MACAROON_TTL_SECONDS
    sig = _sign(payment_hash, expiry, secret)

    raw = f"{payment_hash}:{expiry}:{sig}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_macaroon(token: str) -> tuple[str, int, str]:
    """
    Decode a base64url macaroon token into its (payment_hash, expiry, sig) parts.
    Raises ValueError on any malformed input — caught by the caller.
    """
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
    except Exception as exc:
        raise ValueError(f"Macaroon is not valid base64url: {exc}") from exc

    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Macaroon has wrong number of parts (expected 3, got {len(parts)})")

    payment_hash, expiry_str, sig = parts
    try:
        expiry = int(expiry_str)
    except ValueError as exc:
        raise ValueError(f"Macaroon expiry is not an integer: {expiry_str!r}") from exc

    return payment_hash, expiry, sig


def verify_macaroon(token: str, preimage_hex: str) -> tuple[bool, str]:
    """
    Verify a macaroon + preimage pair.

    Checks (in order — fail fast):
      1. Token decodes cleanly.
      2. Macaroon has not expired.
      3. HMAC signature is valid (token was minted by us).
      4. SHA256(preimage) matches the payment_hash in the macaroon
         (proves the agent actually paid the Lightning invoice).

    Args:
        token:        The base64url macaroon string from the Authorization header.
        preimage_hex: The hex-encoded Lightning payment preimage from the same header.

    Returns:
        (True, "")           — all checks passed, proceed.
        (False, reason_str)  — one check failed, reason_str describes why.
    """
    secret = _get_l402_secret()

    try:
        payment_hash, expiry, provided_sig = _decode_macaroon(token)
    except ValueError as exc:
        return False, f"Malformed macaroon: {exc}"

    # Check 2 — expiry
    now = int(time.time())
    if now > expiry:
        return False, f"Macaroon expired {now - expiry}s ago (TTL={MACAROON_TTL_SECONDS}s)"

    # Check 3 — HMAC (constant-time comparison to prevent timing attacks)
    expected_sig = _sign(payment_hash, expiry, secret)
    if not secrets.compare_digest(provided_sig, expected_sig):
        return False, "Macaroon signature is invalid"

    # Check 4 — preimage proves payment
    try:
        preimage_bytes = bytes.fromhex(preimage_hex)
    except ValueError:
        return False, f"Preimage is not valid hex: {preimage_hex!r}"

    derived_hash = hashlib.sha256(preimage_bytes).hexdigest()
    if not secrets.compare_digest(derived_hash, payment_hash):
        return False, (
            f"Preimage does not match payment_hash. "
            f"SHA256(preimage)={derived_hash[:16]}... "
            f"expected={payment_hash[:16]}..."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Authorization header parser
# ---------------------------------------------------------------------------

def _parse_l402_header(auth_header: str) -> Optional[tuple[str, str]]:
    """
    Parse an L402 Authorization header into (macaroon, preimage).

    Valid format:  "L402 <macaroon_base64url>:<preimage_hex>"

    Returns None if the header is absent, wrong scheme, or malformed.
    The caller decides whether a None here means 402 (no payment yet) or
    401 (payment attempted but header is garbled).
    """
    if not auth_header:
        return None

    scheme, _, credentials = auth_header.partition(" ")
    if scheme.upper() != "L402":
        return None

    # The credentials are exactly "<macaroon>:<preimage>" where preimage is
    # 64 hex chars (SHA256 = 32 bytes = 64 hex digits).  We split on the
    # *last* colon so that the macaroon's base64 (which may contain "=")
    # is not accidentally split.  Preimage is always the last 64 hex chars.
    if ":" not in credentials:
        return None

    # Split on last colon: preimage is always exactly 64 hex chars
    idx = credentials.rfind(":")
    macaroon  = credentials[:idx]
    preimage  = credentials[idx + 1:]

    if not macaroon or not preimage:
        return None

    return macaroon, preimage


# ---------------------------------------------------------------------------
# FastAPI dependency — the L402 gate
# ---------------------------------------------------------------------------

async def l402_gate(request: Request) -> None:
    """
    FastAPI dependency that enforces L402 payment on every protected endpoint.

    Usage:
        @router.post("/v1/quote")
        async def quote(req: QuoteRequest, _: None = Depends(l402_gate)):
            ...

    Flow:
        ┌─ Authorization header present and starts with "L402 "?
        │
        ├─ YES → parse macaroon:preimage → verify_macaroon()
        │         ├─ VALID   → return (allow request to proceed)
        │         └─ INVALID → raise 401 (bad credentials, not a payment demand)
        │
        └─ NO  → create Coinos invoice → mint macaroon
                  └─ raise 402 with WWW-Authenticate challenge

    Error cases:
        - Coinos is unreachable → raise 503 Service Unavailable.
          Returning a 402 when we can't actually generate a payable invoice
          would be misleading — the agent has no invoice to pay.
        - L402_SECRET missing → raise 500 (misconfiguration, not agent's fault).
    """
    auth_header = request.headers.get("Authorization", "")
    parsed = _parse_l402_header(auth_header)

    if parsed is not None:
        # Agent has provided credentials — verify them.
        macaroon, preimage = parsed
        try:
            valid, reason = verify_macaroon(macaroon, preimage)
        except RuntimeError as exc:
            # L402_SECRET misconfigured
            logger.error("L402 secret misconfigured: %s", exc)
            raise HTTPException(status_code=500, detail="Payment system misconfigured") from exc

        if valid:
            logger.debug("L402 verified | macaroon=%s...", macaroon[:12])
            return   # ← proceed to the endpoint

        logger.warning("L402 verification failed: %s", reason)
        raise HTTPException(
            status_code=401,
            detail=f"Invalid L402 credentials: {reason}",
        )

    # No credentials — issue a payment challenge.
    logger.debug("L402 challenge issued for %s %s", request.method, request.url.path)

    try:
        invoice: LightningInvoice = await create_invoice()
    except CoinosError as exc:
        logger.error("Failed to create Coinos invoice: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Payment system temporarily unavailable. "
                "The Lightning invoice provider (Coinos) could not be reached. "
                "Please retry shortly."
            ),
        ) from exc
    except RuntimeError as exc:
        # COINOS_API_KEY missing
        logger.error("Coinos API key not configured: %s", exc)
        raise HTTPException(status_code=500, detail="Payment system misconfigured") from exc

    try:
        macaroon_token = mint_macaroon(invoice.payment_hash)
    except RuntimeError as exc:
        logger.error("Could not mint macaroon: %s", exc)
        raise HTTPException(status_code=500, detail="Payment system misconfigured") from exc

    # RFC 7235 / L402 spec: respond with 402 and WWW-Authenticate header.
    # The header value is intentionally compact — agents parse it, not humans.
    www_auth = f'L402 macaroon="{macaroon_token}", invoice="{invoice.bolt11}"'

    logger.info(
        "L402 challenge sent | amount=%d sats | hash=%s...",
        invoice.amount_sats, invoice.payment_hash[:12],
    )

    raise HTTPException(
        status_code=402,
        headers={"WWW-Authenticate": www_auth},
        detail={
            "error": "Payment required",
            "amount_sats": invoice.amount_sats,
            "invoice": invoice.bolt11,
            "macaroon": macaroon_token,
            "instructions": (
                "Pay the Lightning invoice, then retry this request with: "
                "Authorization: L402 <macaroon>:<preimage>"
            ),
        },
    )
