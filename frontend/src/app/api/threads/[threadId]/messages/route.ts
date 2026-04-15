import { NextRequest, NextResponse } from "next/server";
import {
  appendAiMessage,
  appendHumanMessage,
  getThread,
} from "@/lib/thread-store";
import { callDairyWebhook } from "@/lib/dairy-backend";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

interface PostPayload {
  content: string;
  model: string;
  useTavily: boolean;
}

export async function POST(req: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const auth = getBearerToken(req);
  if (!auth) return unauthorizedResponse();

  const { threadId } = await context.params;
  const body = (await req.json()) as Partial<PostPayload>;

  if (!body?.content || !body.model) {
    return NextResponse.json({ error: "Missing content or model" }, { status: 400 });
  }
  const thread = getThread(threadId);
  if (!thread) {
    return NextResponse.json({ error: "Thread not found" }, { status: 404 });
  }

  const sessionId = `frontend-${threadId}`;
  appendHumanMessage(threadId, body.content);

  try {
    const data = await callDairyWebhook(
      body.model,
      body.content,
      sessionId,
      `Bearer ${auth}`,
    );
    appendAiMessage(threadId, data.response ?? "", body.model);

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
