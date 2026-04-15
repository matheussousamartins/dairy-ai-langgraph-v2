import { NextRequest, NextResponse } from "next/server";
import { backend, buildBackendHeaders, resolveAgentEndpoint } from "@/lib/dairy-backend";
import { appendAiMessage, appendHumanMessage, getThread } from "@/lib/thread-store";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

export const runtime = "nodejs";

interface PostPayload {
  content: string;
  model: string;
  useTavily: boolean;
}

export async function POST(req: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const authToken = getBearerToken(req);
  if (!authToken) return unauthorizedResponse();

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

  // Endpoint de streaming: /webhook/agente-1/stream ou /webhook/orquestrador/stream
  const baseEndpoint = resolveAgentEndpoint(body.model);
  const streamEndpoint = backend(`${baseEndpoint}/stream`);

  const backendRes = await fetch(streamEndpoint, {
    method: "POST",
    headers: buildBackendHeaders(`Bearer ${authToken}`),
    body: JSON.stringify({ message: body.content, session_id: sessionId }),
  });

  if (!backendRes.ok || !backendRes.body) {
    const error = await backendRes.text();
    return NextResponse.json({ error: error || "Backend stream failed" }, { status: backendRes.status });
  }

  const encoder = new TextEncoder();

  // Faz proxy do SSE do backend, interceptando o evento "final"
  // para salvar a resposta no thread-store local
  const stream = new ReadableStream({
    async start(controller) {
      const reader = backendRes.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulatedText = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() ?? "";

          for (const raw of lines) {
            const trimmed = raw.trim();
            if (!trimmed.startsWith("data:")) continue;

            const jsonText = trimmed.slice(trimmed.indexOf("data:") + 5).trim();
            if (!jsonText) continue;

            try {
              const payload = JSON.parse(jsonText) as { event: string; text?: string };
              if (payload.event === "chunk" && payload.text) {
                accumulatedText += payload.text;
              } else if (payload.event === "final") {
                // Salva resposta completa no thread-store
                appendAiMessage(threadId, accumulatedText, body.model!);
              }
            } catch {
              // chunk malformado — ignora e repassa
            }

            // Repassa o chunk SSE original para o browser
            controller.enqueue(encoder.encode(`${trimmed}\n\n`));
          }
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Stream error";
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify({ event: "error", detail })}\n\n`),
        );
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
