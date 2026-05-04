import { NextRequest, NextResponse } from "next/server";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";
import { getTestEvaluationsRepository } from "@/lib/test-evaluations-repository";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ threadId: string }> },
) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  const { threadId } = await params;

  try {
    const repository = getTestEvaluationsRepository();
    return NextResponse.json(await repository.getThreadTestState(threadId));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao carregar o estado de testes.";
    console.error("GET /api/tests/threads/[threadId]:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
