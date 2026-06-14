/**
 * Cross-Chain Route Optimizer — Vercel AI SDK tool definition
 *
 * Submit via PR to: https://github.com/xn1cklas/ai-tools-registry
 *
 * This tool finds the mathematically optimal cross-chain EVM token swap route
 * using True Execution Cost (TEC) analysis.  Requires L402 Lightning payment
 * (~10 sats ≈ $0.006 per call).
 *
 * L402 Payment Flow:
 *   1. Call the tool → receive 402 with WWW-Authenticate header
 *   2. Pay the BOLT11 invoice via Lightning
 *   3. Retry with Authorization: L402 <macaroon>:<preimage>
 */

import { tool } from "ai";
import { z } from "zod";

const OPTIMIZER_BASE_URL =
  process.env.OPTIMIZER_API_URL ?? "https://optimizer-api-crypto.onrender.com";

// ─── Types ───────────────────────────────────────────────────────────────────

interface L402Challenge {
  macaroon: string;
  invoice: string; // BOLT11 string
}

interface OptimalRouteResponse {
  optimal_route: RouteBreakdown | null;
  all_routes: RouteBreakdown[];
  gas_snapshot: GasEntry[];
  meta: Record<string, unknown>;
}

interface RouteBreakdown {
  route_id: string;
  source: "1inch" | "lifi";
  route_type: string;
  chain_in: string;
  chain_out: string;
  token_in: string;
  token_out: string;
  amount_out: number;
  effective_rate: number;
  profit_score: number;
  true_execution_cost_usd: number;
  slippage_usd: number;
  gas_usd: number;
  bridge_fee_usd: number;
  time_penalty_usd: number;
  slippage_bps: number;
  bridge_time_seconds: number;
  is_viable: boolean;
  disqualification_reason: string | null;
}

interface GasEntry {
  chain: string;
  base_fee_gwei: number;
  eth_price_usd: number;
}

// ─── L402 helper ─────────────────────────────────────────────────────────────

function parseL402Header(header: string): L402Challenge {
  // WWW-Authenticate: L402 macaroon=<token>, invoice=<BOLT11>
  const macaroonMatch = header.match(/macaroon=([^,]+)/);
  const invoiceMatch = header.match(/invoice=(.+)$/);
  if (!macaroonMatch || !invoiceMatch) {
    throw new Error(`Malformed WWW-Authenticate header: ${header}`);
  }
  return {
    macaroon: macaroonMatch[1].trim(),
    invoice: invoiceMatch[1].trim(),
  };
}

// ─── Tool definition ─────────────────────────────────────────────────────────

export const crossChainRouteOptimizer = tool({
  description: `
    Find the cheapest cross-chain EVM token swap route using True Execution Cost
    (TEC = slippage + gas + bridge_fee + time_penalty) analysis.

    WHEN TO USE:
    - Before executing any DeFi token swap on Ethereum, Arbitrum, Base, Optimism, or Polygon
    - To compare same-chain DEX routes vs cross-chain bridge routes
    - To get a slippage estimate before signing a transaction

    PAYMENT: This tool costs ~10 satoshis ($0.006) per call via Lightning Network (L402).
    You must provide a 'payInvoice' function that pays a BOLT11 invoice and returns
    the 32-byte preimage hex string.

    SUPPORTED TOKENS: ETH, USDC, USDT, DAI, WETH
    SUPPORTED CHAINS: ethereum, arbitrum, base, optimism, polygon
  `,

  parameters: z.object({
    token_in: z
      .enum(["ETH", "USDC", "USDT", "DAI", "WETH"])
      .describe("Input token ticker symbol"),

    token_out: z
      .enum(["ETH", "USDC", "USDT", "DAI", "WETH"])
      .describe("Output token ticker symbol — must differ from token_in"),

    amount_in: z
      .number()
      .positive()
      .describe("Input amount in token units (e.g. 5.0 for 5 ETH)"),

    amount_in_usd: z
      .number()
      .positive()
      .describe(
        "Current USD value of amount_in — compute from a price oracle before calling"
      ),

    target_price_usd: z
      .number()
      .positive()
      .describe(
        "Spot price of token_out in USD (1.0 for stablecoins, current ETH price for ETH)"
      ),

    chains: z
      .array(
        z.enum(["ethereum", "arbitrum", "base", "optimism", "polygon"])
      )
      .min(1)
      .max(5)
      .describe(
        "Chains to search. Include multiple for cross-chain route discovery."
      ),

    max_slippage_bps: z
      .number()
      .int()
      .min(1)
      .max(2000)
      .default(100)
      .describe("Max slippage in basis points (100 = 1%)"),

    agent_time_value_per_second: z
      .number()
      .min(0)
      .default(0.0001)
      .describe(
        "USD value of 1 second of waiting — used to penalise slow bridge routes"
      ),

    max_bridge_time_seconds: z
      .number()
      .int()
      .min(0)
      .default(900)
      .describe("Hard limit on cross-chain bridge time in seconds"),

    // L402 credentials — populated after paying the invoice
    l402_macaroon: z
      .string()
      .optional()
      .describe("L402 macaroon from previous 402 response (auto-filled)"),

    l402_preimage: z
      .string()
      .optional()
      .describe("Lightning payment preimage hex (auto-filled after payment)"),
  }),

  execute: async (params, { payInvoice }: { payInvoice?: (bolt11: string) => Promise<string> } = {}) => {
    const body = {
      token_in: params.token_in,
      token_out: params.token_out,
      amount_in: params.amount_in,
      amount_in_usd: params.amount_in_usd,
      target_price_usd: params.target_price_usd,
      chains: params.chains,
      max_slippage_bps: params.max_slippage_bps,
      agent_time_value_per_second: params.agent_time_value_per_second,
      max_bridge_time_seconds: params.max_bridge_time_seconds,
    };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (params.l402_macaroon && params.l402_preimage) {
      headers["Authorization"] =
        `L402 ${params.l402_macaroon}:${params.l402_preimage}`;
    }

    const res = await fetch(`${OPTIMIZER_BASE_URL}/v1/quote`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    // L402 payment required
    if (res.status === 402) {
      const wwwAuth = res.headers.get("WWW-Authenticate") ?? "";
      const challenge = parseL402Header(wwwAuth);

      if (!payInvoice) {
        return {
          error: "payment_required",
          invoice: challenge.invoice,
          macaroon: challenge.macaroon,
          instructions:
            "Pay the BOLT11 invoice to receive a preimage, then retry with l402_macaroon and l402_preimage set.",
        };
      }

      // Auto-pay if the agent has Lightning capability
      const preimage = await payInvoice(challenge.invoice);
      const paidRes = await fetch(`${OPTIMIZER_BASE_URL}/v1/quote`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `L402 ${challenge.macaroon}:${preimage}`,
        },
        body: JSON.stringify(body),
      });

      if (!paidRes.ok) {
        throw new Error(`Optimizer API error after payment: ${paidRes.status}`);
      }
      return (await paidRes.json()) as OptimalRouteResponse;
    }

    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`Optimizer API error ${res.status}: ${detail}`);
    }

    return (await res.json()) as OptimalRouteResponse;
  },
});
