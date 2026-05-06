"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { type GenesisMessage } from "@/state/useGenesisUI";
import { useAuth } from "@/state/useAuth";
import { type EvaluationErrorCategory, type EvaluationStatus, type JsonValue } from "@/lib/test-store";

export type TestVerdict = "correct" | "partial" | "incorrect";
export type TestErrorCategory = EvaluationErrorCategory;

interface SaveEvaluationOptions {
  comment?: string;
  errorCategory?: EvaluationErrorCategory;
  expectedAnswer?: string;
}

interface TestSessionMetrics {
  evaluated_count: number;
  correct_count: number;
  partial_count: number;
  incorrect_count: number;
  score_percent: number;
}

interface TestSession {
  id: string;
  thread_id: string;
  title: string;
  status: "active" | "completed";
  created_at: string;
  updated_at: string;
  started_at: string;
  ended_at?: string;
  metrics: TestSessionMetrics;
}

interface TestEvaluation {
  id: string;
  session_id: string;
  thread_id: string;
  message_id: string;
  turn_id?: string;
  verdict: TestVerdict;
  score: number;
  question: string;
  answer: string;
  agent_id?: string;
  model_id?: string;
  comment?: string;
  evaluator_id?: string;
  environment?: string;
  app_version?: string;
  git_sha?: string;
  rag_architecture?: string;
  prompt_version?: string;
  retrieval_config_version?: string;
  error_category?: EvaluationErrorCategory;
  expected_answer?: string;
  status?: EvaluationStatus;
  answer_source?: string;
  chosen_agent_ids?: number[];
  primary_agent_id?: string;
  top_rag_score?: number;
  rag_sources?: string[];
  rag_search_count?: number;
  node_count?: number;
  latency_ms?: number;
  web_fallback_used?: boolean;
  metadata?: Record<string, JsonValue>;
  created_at: string;
  updated_at: string;
}

interface ThreadTestStateResponse {
  session: TestSession | null;
  evaluations: TestEvaluation[];
  evaluations_by_message_id: Record<string, TestEvaluation>;
}

function getThreadTitle(threadId: string, messages: GenesisMessage[]) {
  const firstUser = messages.find((message) => message.role === "user")?.content ?? "";
  return firstUser ? firstUser.slice(0, 42) : `Thread ${threadId.slice(0, 8)}`;
}

function findPreviousUserQuestion(messages: GenesisMessage[], assistantMessageId: string) {
  const assistantIndex = messages.findIndex((message) => message.id === assistantMessageId);
  if (assistantIndex <= 0) return "";
  for (let index = assistantIndex - 1; index >= 0; index -= 1) {
    const candidate = messages[index];
    if (candidate?.role === "user") {
      return candidate.content;
    }
  }
  return "";
}

function uniqueValues<T>(values: T[]) {
  return [...new Set(values.filter(Boolean))];
}

function parseTraceChunks(output?: string) {
  if (!output) return [] as Array<{ content: string; score: number | null; source: string }>;
  try {
    const parsed = JSON.parse(output) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((item) => {
        if (!item || typeof item !== "object") return null;
        const row = item as { content?: unknown; score?: unknown; source?: unknown };
        const score = typeof row.score === "number" ? row.score : null;
        return {
          content: typeof row.content === "string" ? row.content : "",
          score,
          source: typeof row.source === "string" ? row.source : "",
        };
      })
      .filter((item): item is { content: string; score: number | null; source: string } => Boolean(item));
  } catch {
    return [];
  }
}

function extractAgentIdsFromTrace(message: GenesisMessage) {
  const ids = new Set<number>();
  message.trace?.forEach((event) => {
    const label = `${event.tool ?? ""} ${event.node ?? ""}`;
    const baseMatch = label.match(/base[_-](\d+)/i);
    if (baseMatch?.[1]) ids.add(Number(baseMatch[1]));
    if (/regulat/i.test(label)) ids.add(3);
    if (/especialista/i.test(label)) ids.add(1);
  });
  return [...ids].filter((item) => Number.isFinite(item));
}

function buildEvaluationQualitySignals(message: GenesisMessage, threadId: string, question: string) {
  const trace = message.trace ?? [];
  const nodeNames = uniqueValues(trace.map((event) => event.node).filter((node): node is string => Boolean(node)));
  const toolNames = uniqueValues(trace.map((event) => event.tool).filter((tool): tool is string => Boolean(tool)));
  const ragEvents = trace.filter((event) => event.type === "rag_result" || event.type === "tool_result");
  const chunks = ragEvents.flatMap((event) => parseTraceChunks(event.output));
  const sources = uniqueValues(chunks.map((chunk) => chunk.source).filter(Boolean));
  const scores = chunks
    .map((chunk) => chunk.score)
    .filter((score): score is number => typeof score === "number" && Number.isFinite(score));
  const topRagScore = scores.length > 0 ? Math.max(...scores) : undefined;
  const ragToolCalls = trace.filter((event) => event.type === "tool_call" && /buscar_base|rag|knowledge|web|tavily/i.test(event.tool ?? ""));
  const ragSearchCount = Math.max(ragToolCalls.length, ragEvents.length);
  const nodeCount = trace.filter((event) => event.type === "node_start").length;
  const timestamps = trace.map((event) => event.ts).filter((ts) => Number.isFinite(ts));
  const latencyMs = timestamps.length >= 2 ? Math.max(...timestamps) - Math.min(...timestamps) : undefined;
  const webFallbackUsed =
    Boolean(message.usedTavily) ||
    trace.some((event) => /web|internet|tavily|fallback/i.test(`${event.tool ?? ""} ${event.node ?? ""}`));
  const chosenAgentIds = extractAgentIdsFromTrace(message);
  const answerSource = webFallbackUsed ? "web_fallback" : ragSearchCount > 0 ? "rag" : nodeNames.includes("respond_direct") ? "direct" : "unknown";

  const metadata: Record<string, JsonValue> = {
    thread_id: threadId,
    message_id: message.id,
    turn_id: message.turnId ?? null,
    assistant_agent_id: message.agentId ?? null,
    assistant_model_id: message.modelId ?? null,
    message_timestamp: message.timestamp,
    question_length: question.length,
    answer_length: message.content.length,
    trace_available: trace.length > 0,
    trace_summary: {
      event_count: trace.length,
      node_names: nodeNames,
      tool_names: toolNames,
      rag_event_count: ragEvents.length,
      tool_call_count: trace.filter((event) => event.type === "tool_call").length,
      first_ts: timestamps.length ? Math.min(...timestamps) : null,
      last_ts: timestamps.length ? Math.max(...timestamps) : null,
    },
    rag_chunks_preview: chunks.slice(0, 12).map((chunk) => ({
      source: chunk.source || null,
      score: chunk.score,
      content_preview: chunk.content.slice(0, 240),
    })),
  };

  return {
    metadata,
    answerSource,
    chosenAgentIds,
    primaryAgentId: message.agentId,
    topRagScore,
    ragSources: sources,
    ragSearchCount,
    nodeCount,
    latencyMs,
    webFallbackUsed,
  };
}

export function useThreadTesting(threadId: string | null, messages: GenesisMessage[]) {
  const { token } = useAuth();
  const [session, setSession] = useState<TestSession | null>(null);
  const [evaluationsByMessageId, setEvaluationsByMessageId] = useState<Record<string, TestEvaluation>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchState = useCallback(async () => {
    if (!threadId || !token) {
      setSession(null);
      setEvaluationsByMessageId({});
      setErrorMessage(null);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`/api/tests/threads/${threadId}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: "no-store",
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        setErrorMessage(payload?.error ?? "Falha ao carregar o estado de testes.");
        return;
      }

      const data = (await response.json()) as ThreadTestStateResponse;
      setSession(data.session);
      setEvaluationsByMessageId(data.evaluations_by_message_id ?? {});
      setErrorMessage(null);
    } finally {
      setIsLoading(false);
    }
  }, [threadId, token]);

  useEffect(() => {
    fetchState().catch(console.error);
  }, [fetchState]);

  const saveEvaluation = useCallback(
    async (message: GenesisMessage, verdict: TestVerdict, options?: SaveEvaluationOptions | string) => {
      if (!threadId || !token) return false;

      const question = findPreviousUserQuestion(messages, message.id);
      const normalizedOptions = typeof options === "string" ? { comment: options } : options ?? {};
      const qualitySignals = buildEvaluationQualitySignals(message, threadId, question);
      setIsSaving(true);
      try {
        const response = await fetch(`/api/tests/threads/${threadId}/evaluate`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            threadTitle: getThreadTitle(threadId, messages),
            messageId: message.id,
            turnId: message.turnId,
            verdict,
            question,
            answer: message.content,
            agentId: message.agentId,
            modelId: message.modelId,
            comment: normalizedOptions.comment,
            errorCategory: verdict === "correct" ? undefined : normalizedOptions.errorCategory,
            expectedAnswer: normalizedOptions.expectedAnswer,
            ...qualitySignals,
          }),
        });

        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          setErrorMessage(payload?.error ?? "Falha ao salvar avaliação.");
          return false;
        }

        const data = (await response.json()) as { session: TestSession; evaluation: TestEvaluation };
        setSession(data.session);
        setEvaluationsByMessageId((prev) => ({ ...prev, [data.evaluation.message_id]: data.evaluation }));
        setErrorMessage(null);
        return true;
      } catch (error) {
        const messageText =
          error instanceof Error ? error.message : "Falha ao salvar avaliação.";
        setErrorMessage(messageText);
        console.error("saveEvaluation:", error);
        return false;
      } finally {
        setIsSaving(false);
      }
    },
    [messages, threadId, token],
  );

  const evaluateMessage = useCallback(
    async (message: GenesisMessage, verdict: TestVerdict) => {
      const existing = evaluationsByMessageId[message.id];
      return saveEvaluation(message, verdict, {
        comment: existing?.comment,
        errorCategory: existing?.error_category,
        expectedAnswer: existing?.expected_answer,
      });
    },
    [evaluationsByMessageId, saveEvaluation],
  );

  const updateEvaluationComment = useCallback(
    async (message: GenesisMessage, comment: string) => {
      const existing = evaluationsByMessageId[message.id];
      if (!existing?.verdict) return false;
      return saveEvaluation(message, existing.verdict, {
        comment: comment.trim() || undefined,
        errorCategory: existing.error_category,
        expectedAnswer: existing.expected_answer,
      });
    },
    [evaluationsByMessageId, saveEvaluation],
  );

  const finalizeSession = useCallback(async () => {
    if (!session?.id || !token) return false;

    setIsSaving(true);
    try {
      const response = await fetch(`/api/tests/sessions/${session.id}/finalize`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { error?: string } | null;
        setErrorMessage(payload?.error ?? "Falha ao finalizar sessão de teste.");
        return false;
      }

      const data = (await response.json()) as { session: TestSession };
      setSession(data.session);
      setErrorMessage(null);
      return true;
    } catch (error) {
      const messageText =
        error instanceof Error ? error.message : "Falha ao finalizar sessão de teste.";
      setErrorMessage(messageText);
      console.error("finalizeSession:", error);
      return false;
    } finally {
      setIsSaving(false);
    }
  }, [session?.id, token]);

  const assistantMessageCount = useMemo(
    () =>
      messages.filter(
        (message) =>
          message.role === "assistant" &&
          message.content.trim() &&
          message.content !== "Pensando..." &&
          message.content !== "Falha ao gerar resposta. Tente novamente." &&
          findPreviousUserQuestion(messages, message.id).trim().length >= 15,
      ).length,
    [messages],
  );

  return {
    session,
    evaluationsByMessageId,
    isLoading,
    isSaving,
    errorMessage,
    assistantMessageCount,
    saveEvaluation,
    evaluateMessage,
    updateEvaluationComment,
    finalizeSession,
    clearErrorMessage: () => setErrorMessage(null),
    refreshThreadTesting: fetchState,
  };
}
