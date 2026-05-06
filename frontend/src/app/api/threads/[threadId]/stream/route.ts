import { NextRequest, NextResponse } from "next/server";
import { backend, buildBackendHeaders, resolveAgentEndpoint } from "@/lib/dairy-backend";
import {
  appendAiMessage,
  appendHumanMessage,
  appendTrace,
  deriveConsoleThreadOwnerId,
  getThread,
} from "@/lib/thread-store";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

export const runtime = "nodejs";

const DEFAULT_TYPING_SPLIT_THRESHOLD = 48;
const DEFAULT_TYPING_MAX_SEGMENT = 14;
const DEFAULT_TYPING_DELAY_MS = 18;

function readPositiveIntEnv(name: string, fallback: number, min: number, max: number) {
  const rawValue = process.env[name];
  const parsed = Number.parseInt(rawValue ?? "", 10);

  if (!Number.isFinite(parsed)) {
    return fallback;
  }

  return Math.min(Math.max(parsed, min), max);
}

const TYPING_SPLIT_THRESHOLD = readPositiveIntEnv(
  "STREAM_FALLBACK_SPLIT_THRESHOLD",
  DEFAULT_TYPING_SPLIT_THRESHOLD,
  12,
  240,
);
const TYPING_MAX_SEGMENT = readPositiveIntEnv(
  "STREAM_FALLBACK_MAX_SEGMENT",
  DEFAULT_TYPING_MAX_SEGMENT,
  4,
  80,
);
const TYPING_DELAY_MS = readPositiveIntEnv(
  "STREAM_FALLBACK_DELAY_MS",
  DEFAULT_TYPING_DELAY_MS,
  0,
  250,
);

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
  const authToken = getBearerToken(req);
  if (!authToken) return unauthorizedResponse();
  const ownerId = deriveConsoleThreadOwnerId(authToken);

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

  const baseEndpoint = resolveAgentEndpoint(body.agentId);
  const streamEndpoint = backend(`${baseEndpoint}/stream`);

  const backendRes = await fetch(streamEndpoint, {
    method: "POST",
    headers: buildBackendHeaders(`Bearer ${authToken}`),
    body: JSON.stringify({ message: body.content, session_id: sessionId, model: body.model }),
  });

  if (!backendRes.ok || !backendRes.body) {
    const error = await backendRes.text();
    return NextResponse.json({ error: error || "Backend stream failed" }, { status: backendRes.status });
  }

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const reader = backendRes.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulatedText = "";
      let savedAssistant = false;
      let savedTrace = false;
      const traceEvents: Array<{
        type: "node_start" | "node_end" | "tool_call" | "tool_result" | "rag_result";
        node?: string;
        tool?: string;
        input?: string;
        output?: string;
        ts: number;
      }> = [];

      const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

      const splitForTyping = (text: string) => {
        if (text.length <= TYPING_SPLIT_THRESHOLD) return [text];

        const tokens = text.match(/\S+\s*|\s+/g) ?? [text];
        const segments: string[] = [];
        let current = "";

        for (const token of tokens) {
          if ((current + token).length > TYPING_MAX_SEGMENT && current) {
            segments.push(current);
            current = token;
          } else {
            current += token;
          }
        }

        if (current) segments.push(current);
        return segments;
      };

      const emitChunkText = async (text: string) => {
        const segments = splitForTyping(text);
        for (let index = 0; index < segments.length; index += 1) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ event: "chunk", text: segments[index] })}\n\n`),
          );
          if (segments.length > 1 && index < segments.length - 1) {
            await sleep(TYPING_DELAY_MS);
          }
        }
      };

      const processSseRecord = async (trimmed: string) => {
        if (!trimmed.startsWith("data:")) return;

        const jsonText = trimmed.slice(trimmed.indexOf("data:") + 5).trim();
        if (!jsonText) return;

        try {
          const payload = JSON.parse(jsonText) as {
            event: string;
            text?: string;
            type?: "node_start" | "node_end" | "tool_call" | "tool_result" | "rag_result";
            node?: string;
            tool?: string;
            input?: string;
            output?: string;
            ts?: number;
          };

          if (payload.event === "chunk" && payload.text) {
            accumulatedText += payload.text;
            await emitChunkText(payload.text);
            return;
          } else if (payload.event === "trace" && payload.type) {
            traceEvents.push({
              type: payload.type,
              node: payload.node,
              tool: payload.tool,
              input: payload.input,
              output: payload.output,
              ts: payload.ts ?? Date.now(),
            });
          } else if (payload.event === "final") {
            await appendAiMessage(ownerId, threadId, accumulatedText, body.model!, body.agentId, undefined, turnId);
            savedAssistant = true;
            if (traceEvents.length > 0) {
              await appendTrace(ownerId, threadId, turnId, traceEvents, body.model!, body.agentId);
              savedTrace = true;
            }
          }
        } catch {
          // chunk malformado — ignora e repassa
        }

        controller.enqueue(encoder.encode(`${trimmed}\n\n`));
      };

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const records = buffer.split("\n\n");
          buffer = records.pop() ?? "";

          for (const raw of records) {
            await processSseRecord(raw.trim());
          }
        }

        const trailing = buffer.trim();
        if (trailing) {
          await processSseRecord(trailing);
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : "Stream error";
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ event: "error", detail })}\n\n`));
      } finally {
        if (!savedAssistant && accumulatedText.trim()) {
          await appendAiMessage(ownerId, threadId, accumulatedText, body.model!, body.agentId, undefined, turnId);
        }
        if (!savedTrace && traceEvents.length > 0) {
          await appendTrace(ownerId, threadId, turnId, traceEvents, body.model!, body.agentId);
        }
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
