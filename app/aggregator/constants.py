"""
aggregator/constants.py
-----------------------
Shared lookup tables used by every aggregator module.

Rules:
  - Add new chains here first, then update the Chain enum in models.py.
  - Token addresses are checksummed EIP-55 strings. Never lowercase them —
    some APIs reject non-checksummed addresses.
  - All decimals are integer exponents (USDC = 6 means divide raw by 10^6).
"""

from __future__ import annotations
from typing import Dict
from app.optimizer.models import Chain


# ---------------------------------------------------------------------------
# Chain → numeric ID (EVM chainId)
# ---------------------------------------------------------------------------
CHAIN_ID: Dict[Chain, int] = {
    Chain.ETHEREUM: 1,
    Chain.ARBITRUM: 42161,
    Chain.BASE:     8453,
    Chain.OPTIMISM: 10,
    Chain.POLYGON:  137,
}

# Reverse map: numeric chainId → Chain enum (used when parsing API responses)
CHAIN_FROM_ID: Dict[int, Chain] = {v: k for k, v in CHAIN_ID.items()}


# ---------------------------------------------------------------------------
# Public JSON-RPC endpoints (no key required, rate-limit tolerant)
# ---------------------------------------------------------------------------
# All are llamarpc / official L2 RPCs that support eth_feeHistory.
PUBLIC_RPC: Dict[Chain, str] = {
    Chain.ETHEREUM: "https://eth.llamarpc.com",
    Chain.ARBITRUM: "https://arb1.arbitrum.io/rpc",
    Chain.BASE:     "https://mainnet.base.org",
    Chain.OPTIMISM: "https://mainnet.optimism.io",
    Chain.POLYGON:  "https://polygon-rpc.com",
}


# ---------------------------------------------------------------------------
# Token addresses (checksummed)
# ---------------------------------------------------------------------------
# The 1inch convention for native ETH (not WETH) on any EVM chain:
NATIVE_ETH_ADDR = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEEE"

# Per-chain token address lookup.  Keys are uppercase ticker symbols.
# Aggregators that require contract addresses (1inch) read from here.
TOKEN_ADDRESS: Dict[str, Dict[Chain, str]] = {
    "ETH": {
        Chain.ETHEREUM: NATIVE_ETH_ADDR,
        Chain.ARBITRUM: NATIVE_ETH_ADDR,
        Chain.BASE:     NATIVE_ETH_ADDR,
        Chain.OPTIMISM: NATIVE_ETH_ADDR,
    },
    "WETH": {
        Chain.ETHEREUM: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        Chain.ARBITRUM: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        Chain.BASE:     "0x4200000000000000000000000000000000000006",
        Chain.OPTIMISM: "0x4200000000000000000000000000000000000006",
    },
    "USDC": {
        # Native USDC (Circle's canonical deployment — not bridged USDC.e)
        Chain.ETHEREUM: "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        Chain.ARBITRUM: "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        Chain.BASE:     "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        Chain.OPTIMISM: "0x0b2C639c533813f4Aa9D7837CAf62653d097Ff85",
        Chain.POLYGON:  "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    },
    "USDT": {
        Chain.ETHEREUM: "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        Chain.ARBITRUM: "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
        Chain.OPTIMISM: "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
        Chain.POLYGON:  "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
    },
    "DAI": {
        Chain.ETHEREUM: "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        Chain.ARBITRUM: "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
        Chain.BASE:     "0x50c5725949A6F0c72E6C4a641F24049A917DB0Cb",
        Chain.OPTIMISM: "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
    },
}

# Token decimal places (used to convert raw integer amounts to float)
TOKEN_DECIMALS: Dict[str, int] = {
    "ETH":  18,
    "WETH": 18,
    "USDC": 6,
    "USDT": 6,
    "DAI":  18,
}

# Default decimal fallback when token is not in the table above
DEFAULT_DECIMALS = 18


# ---------------------------------------------------------------------------
# Li.Fi chain key strings  (used in Li.Fi API params, not the same as chainId)
# ---------------------------------------------------------------------------
LIFI_CHAIN_KEY: Dict[Chain, str] = {
    Chain.ETHEREUM: "ETH",
    Chain.ARBITRUM: "ARB",
    Chain.BASE:     "BAS",
    Chain.OPTIMISM: "OPT",
    Chain.POLYGON:  "POL",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def token_address(ticker: str, chain: Chain) -> str:
    """
    Look up the checksummed contract address for a ticker on a given chain.
    Raises KeyError with a descriptive message if the pair is unsupported —
    fail loud rather than sending a zero address to a DEX.
    """
    ticker = ticker.upper()
    chain_map = TOKEN_ADDRESS.get(ticker)
    if chain_map is None:
        raise KeyError(f"Token '{ticker}' is not in the address table. Add it to constants.py.")
    addr = chain_map.get(chain)
    if addr is None:
        raise KeyError(
            f"Token '{ticker}' has no address entry for chain '{chain.value}'. "
            "Add it to TOKEN_ADDRESS in constants.py."
        )
    return addr


def token_decimals(ticker: str) -> int:
    """Return decimal places for a token, defaulting to 18 if unknown."""
    return TOKEN_DECIMALS.get(ticker.upper(), DEFAULT_DECIMALS)


def raw_to_float(raw_amount: int | str, ticker: str) -> float:
    """
    Convert a raw integer token amount (as returned by APIs) to a float.
    e.g. raw_to_float(5_000_000, "USDC") → 5.0
         raw_to_float("5000000000000000000", "ETH") → 5.0
    """
    decimals = token_decimals(ticker)
    return int(raw_amount) / (10 ** decimals)
