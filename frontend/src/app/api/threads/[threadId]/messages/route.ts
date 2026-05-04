import { NextRequest, NextResponse } from "next/server";
import {
  appendAiMessage,
  appendHumanMessage,
  deriveConsoleThreadOwnerId,
  getThread,
} from "@/lib/thread-store";
import { callDairyWebhook } from "@/lib/dairy-backend";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

interface PostPayload {
  content: string;
  model: string;
  agentId: string;
  useTavily: boolean;
}

export async function POST(req: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const auth = getBearerToken(req);
  if (!auth) return unauthorizedResponse();
  const ownerId = deriveConsoleThreadOwnerId(auth);

  const { threadId } = await context.params;
  const body = (await req.json()) as Partial<PostPayload>;

  if (!body?.content || !body.model || !body.agentId) {
    return NextResponse.json({ error: "Missing content, model or agentId" }, { status: 400 });
  }
  const thread = await getThread(ownerId, threadId);
  if (!thread) {
    return NextResponse.json({ error: "Thread not found" }, { status: 404 });
  }

  const sessionId = `frontend-${threadId}`;
  const turnId = `turn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  await appendHumanMessage(ownerId, threadId, body.content, turnId);

  try {
    const data = await callDairyWebhook(
      body.agentId,
      body.content,
      sessionId,
      body.model,
      `Bearer ${auth}`,
    );
    await appendAiMessage(ownerId, threadId, data.response ?? "", body.model, body.agentId, undefined, turnId);

    return NextResponse.json({
      result: {
        messages: thread.values.messages,
      },
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Failed to send message";
    return NextResponse.json({ error: detail }, { status: 500 });
  }
}
