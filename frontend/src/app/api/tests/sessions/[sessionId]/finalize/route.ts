import { NextRequest, NextResponse } from "next/server";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";
import { getTestEvaluationsRepository } from "@/lib/test-evaluations-repository";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  const { sessionId } = await params;

  try {
    const repository = getTestEvaluationsRepository();
    const session = await repository.finalizeSession(sessionId);
    if (!session) {
      return NextResponse.json({ error: "Sessão de teste não encontrada." }, { status: 404 });
    }

    return NextResponse.json({ session });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao finalizar a sessão de teste.";
    console.error("POST /api/tests/sessions/[sessionId]/finalize:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
