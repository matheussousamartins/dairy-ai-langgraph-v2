"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { type GenesisMessage } from "@/state/useGenesisUI";
import { useAuth } from "@/state/useAuth";

export type TestVerdict = "correct" | "partial" | "incorrect";

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
    async (message: GenesisMessage, verdict: TestVerdict, comment?: string) => {
      if (!threadId || !token) return false;

      const question = findPreviousUserQuestion(messages, message.id);
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
            comment,
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
      return saveEvaluation(message, verdict, evaluationsByMessageId[message.id]?.comment);
    },
    [evaluationsByMessageId, saveEvaluation],
  );

  const updateEvaluationComment = useCallback(
    async (message: GenesisMessage, comment: string) => {
      const existingVerdict = evaluationsByMessageId[message.id]?.verdict;
      if (!existingVerdict) return false;
      return saveEvaluation(message, existingVerdict, comment.trim() || undefined);
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
