import { NextRequest, NextResponse } from "next/server";
import { AGENT_CATALOG } from "@/lib/agent-catalog";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  return NextResponse.json({ agents: AGENT_CATALOG });
}
