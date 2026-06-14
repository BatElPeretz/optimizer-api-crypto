"""
mcp/server.py
-------------
Model Context Protocol (MCP) server for the Cross-Chain Route Optimizer.

This server wraps the HTTP API so that any MCP-compatible client
(Claude Desktop, Cursor, Windsurf, Zed, etc.) can call the optimizer
as a native tool — without the user writing any code.

The L402 payment flow is handled transparently:
  - If the user has pre-configured OPTIMIZER_MACAROON + OPTIMIZER_PREIMAGE,
    it sends them automatically.
  - Otherwise, it returns the Lightning invoice so the user/agent can pay
    and retry.

Installation (Claude Desktop):
  1. pip install mcp httpx
  2. Add to ~/Library/Application Support/Claude/claude_desktop_config.json:
     {
       "mcpServers": {
         "cross-chain-optimizer": {
           "command": "python",
           "args": ["/path/to/optimizer-api/mcp/server.py"],
           "env": {
             "OPTIMIZER_API_URL": "https://optimizer-api-crypto.onrender.com",
             "OPTIMIZER_MACAROON": "optional-pre-paid-macaroon",
             "OPTIMIZER_PREIMAGE": "optional-preimage-hex"
           }
         }
       }
     }
  3. Restart Claude Desktop.

Claude will now have a "get_optimal_swap_route" tool available in every conversation.

Running as standalone MCP server:
    python mcp/server.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import httpx
import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.environ.get("OPTIMIZER_API_URL", "https://optimizer-api-crypto.onrender.com").rstrip("/")
MACAROON = os.environ.get("OPTIMIZER_MACAROON", "")
PREIMAGE = os.environ.get("OPTIMIZER_PREIMAGE", "")

SUPPORTED_CHAINS = ["ethereum", "arbitrum", "base", "optimism", "polygon"]
SUPPORTED_TOKENS = ["ETH", "USDC", "USDT", "DAI", "WETH"]

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("cross-chain-route-optimizer")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_optimal_swap_route",
            description=(
                "Find the mathematically optimal cross-chain EVM token swap route "
                "using True Execution Cost (TEC) analysis.\n\n"
                "TEC = slippage_cost + gas_cost + bridge_fee + time_penalty\n\n"
                "Use this tool when the user wants to:\n"
                "- Execute a DeFi token swap on Ethereum, Arbitrum, Base, Optimism, or Polygon\n"
                "- Compare same-chain DEX routes vs cross-chain bridge routes (1inch vs Li.Fi)\n"
                "- Get a slippage estimate before signing a transaction\n"
                "- Find the cheapest route across multiple chains simultaneously\n\n"
                f"Supported tokens: {', '.join(SUPPORTED_TOKENS)}\n"
                f"Supported chains: {', '.join(SUPPORTED_CHAINS)}\n\n"
                "Cost: ~10 satoshis (~$0.006) per query via Lightning Network (L402). "
                "If not pre-configured, returns a BOLT11 invoice to pay."
            ),
            inputSchema={
                "type": "object",
                "required": [
                    "token_in", "token_out",
                    "amount_in", "amount_in_usd",
                    "target_price_usd", "chains",
                ],
                "properties": {
                    "token_in": {
                        "type": "string",
                        "enum": SUPPORTED_TOKENS,
                        "description": "Input token ticker symbol",
                    },
                    "token_out": {
                        "type": "string",
                        "enum": SUPPORTED_TOKENS,
                        "description": "Output token ticker symbol — must differ from token_in",
                    },
                    "amount_in": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Input amount in token units (e.g. 5.0 for 5 ETH)",
                    },
                    "amount_in_usd": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": (
                            "USD value of amount_in at current market price. "
                            "Fetch from a price oracle before calling."
                        ),
                    },
                    "target_price_usd": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": (
                            "Spot price of token_out in USD. "
                            "Use 1.0 for stablecoins (USDC, USDT, DAI)."
                        ),
                    },
                    "chains": {
                        "type": "array",
                        "items": {"type": "string", "enum": SUPPORTED_CHAINS},
                        "minItems": 1,
                        "maxItems": 5,
                        "description": (
                            "Chains to search. Include multiple for cross-chain discovery. "
                            "Example: ['arbitrum', 'base']"
                        ),
                    },
                    "max_slippage_bps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 2000,
                        "default": 100,
                        "description": "Max acceptable slippage in basis points (100 = 1%)",
                    },
                    "max_bridge_time_seconds": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 900,
                        "description": "Hard limit on bridge settlement time in seconds (default 15 min)",
                    },
                    "agent_time_value_per_second": {
                        "type": "number",
                        "minimum": 0,
                        "default": 0.0001,
                        "description": "USD value of 1 second of waiting (used to penalise slow bridges)",
                    },
                },
                "additionalProperties": False,
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    if name != "get_optimal_swap_route":
        raise ValueError(f"Unknown tool: {name}")

    result = await _call_optimizer(arguments)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# HTTP + L402 logic
# ---------------------------------------------------------------------------

async def _call_optimizer(params: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in params.items() if v is not None}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if MACAROON and PREIMAGE:
        headers["Authorization"] = f"L402 {MACAROON}:{PREIMAGE}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(f"{API_URL}/v1/quote", json=body, headers=headers)

        # L402 — payment required
        if resp.status_code == 402:
            www_auth = resp.headers.get("WWW-Authenticate", "")
            macaroon, invoice = _parse_l402_header(www_auth)
            return {
                "status": "payment_required",
                "message": (
                    "This query costs ~10 satoshis (~$0.006) via Lightning Network. "
                    "Pay the invoice below, then provide the preimage to complete the request."
                ),
                "lightning_invoice": invoice,
                "macaroon": macaroon,
                "instructions": (
                    "1. Pay the lightning_invoice using any Lightning wallet.\n"
                    "2. Copy the 64-char preimage hex your wallet returns.\n"
                    "3. Set OPTIMIZER_MACAROON and OPTIMIZER_PREIMAGE env vars.\n"
                    "4. Restart the MCP server and retry."
                ),
            }

        if resp.status_code == 401:
            return {
                "status": "auth_error",
                "message": "Macaroon expired or preimage incorrect. Clear the env vars and restart to get a fresh invoice.",
            }

        resp.raise_for_status()
        data = resp.json()

        # Surface the most actionable fields at the top level for the LLM
        optimal = data.get("optimal_route")
        return {
            "status": "ok",
            "optimal_route": optimal,
            "summary": _summarise(optimal, data.get("meta", {})),
            "all_routes_count": len(data.get("all_routes", [])),
            "viable_routes": data.get("meta", {}).get("viable_routes", 0),
            "latency_ms": data.get("meta", {}).get("latency_ms"),
            "full_response": data,
        }


def _parse_l402_header(header: str) -> tuple[str, str]:
    """Extract macaroon and invoice from WWW-Authenticate: L402 ... header."""
    macaroon = ""
    invoice = ""
    if "macaroon=" in header:
        part = header.split("macaroon=", 1)[1]
        macaroon = part.split(",")[0].strip()
    if "invoice=" in header:
        invoice = header.split("invoice=", 1)[1].strip()
    return macaroon, invoice


def _summarise(route: dict | None, meta: dict) -> str:
    if not route:
        warning = meta.get("warning", "No viable routes found.")
        return f"No optimal route found. {warning}"
    return (
        f"Best route: {route['source']} | "
        f"{route['chain_in']} → {route['chain_out']} | "
        f"Out: {route['amount_out']:.4f} {route['token_out']} | "
        f"TEC: ${route['true_execution_cost_usd']:.4f} | "
        f"Slippage: {route['slippage_bps']} bps | "
        f"Bridge time: {route['bridge_time_seconds']}s"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="cross-chain-route-optimizer",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
