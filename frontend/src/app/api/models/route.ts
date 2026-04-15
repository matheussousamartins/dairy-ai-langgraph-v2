import { NextRequest, NextResponse } from "next/server";
import { AGENT_CATALOG } from "@/lib/agent-catalog";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  const models = AGENT_CATALOG.map((agent) => ({
    id: agent.id,
    label: agent.label,
    input_cost: agent.input_cost ?? 0,
    output_cost: agent.output_cost ?? 0,
  }));

  return NextResponse.json({ models });
}
