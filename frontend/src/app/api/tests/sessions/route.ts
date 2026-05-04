import { NextRequest, NextResponse } from "next/server";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";
import { getTestEvaluationsRepository } from "@/lib/test-evaluations-repository";

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  try {
    const repository = getTestEvaluationsRepository();
    return NextResponse.json({ sessions: await repository.listSessions() });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao listar sessões de teste.";
    console.error("GET /api/tests/sessions:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
