import { NextRequest, NextResponse } from "next/server";
import { ensureAuthorized, getBearerToken, unauthorizedResponse, verifyConsoleToken } from "@/lib/server-auth";
import { type EvaluationErrorCategory, type JsonValue, type TestVerdict } from "@/lib/test-store";
import { getTestEvaluationsRepository } from "@/lib/test-evaluations-repository";

interface EvaluateRequestBody {
  threadTitle?: string;
  messageId?: string;
  turnId?: string;
  verdict?: TestVerdict;
  question?: string;
  answer?: string;
  agentId?: string;
  modelId?: string;
  comment?: string;
  metadata?: Record<string, JsonValue>;
  errorCategory?: EvaluationErrorCategory;
  expectedAnswer?: string;
  answerSource?: string;
  chosenAgentIds?: number[];
  primaryAgentId?: string;
  topRagScore?: number;
  ragSources?: string[];
  ragSearchCount?: number;
  nodeCount?: number;
  latencyMs?: number;
  webFallbackUsed?: boolean;
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ threadId: string }> },
) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  const body = (await req.json()) as EvaluateRequestBody;
  const { threadId } = await params;

  if (!body.messageId || !body.verdict || !body.question || !body.answer) {
    return NextResponse.json({ error: "Payload de avaliação incompleto." }, { status: 400 });
  }

  const VALID_VERDICTS: TestVerdict[] = ["correct", "partial", "incorrect"];
  if (!VALID_VERDICTS.includes(body.verdict)) {
    return NextResponse.json({ error: "Verdict inválido. Use: correct, partial ou incorrect." }, { status: 400 });
  }

  try {
    const tokenPayload = verifyConsoleToken(getBearerToken(req) ?? "");
    const repository = getTestEvaluationsRepository();
    const result = await repository.upsertEvaluation({
      threadId,
      threadTitle: body.threadTitle ?? `Thread ${threadId.slice(0, 8)}`,
      messageId: body.messageId,
      turnId: body.turnId,
      verdict: body.verdict,
      question: body.question,
      answer: body.answer,
      agentId: body.agentId,
      modelId: body.modelId,
      comment: body.comment,
      evaluatorId: tokenPayload?.sub,
      metadata: body.metadata,
      errorCategory: body.errorCategory,
      expectedAnswer: body.expectedAnswer,
      answerSource: body.answerSource,
      chosenAgentIds: body.chosenAgentIds,
      primaryAgentId: body.primaryAgentId,
      topRagScore: body.topRagScore,
      ragSources: body.ragSources,
      ragSearchCount: body.ragSearchCount,
      nodeCount: body.nodeCount,
      latencyMs: body.latencyMs,
      webFallbackUsed: body.webFallbackUsed,
    });

    return NextResponse.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Falha ao salvar avaliação.";
    console.error("POST /api/tests/threads/[threadId]/evaluate:", error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
