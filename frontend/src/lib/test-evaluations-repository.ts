import {
  ensureTestSessionForThread,
  finalizeTestSession as finalizeInMemoryTestSession,
  getThreadTestState as getInMemoryThreadTestState,
  listTestSessions as listInMemoryTestSessions,
  type StoreTestEvaluation,
  type StoreTestSession,
  type TestVerdict,
  upsertThreadEvaluation,
} from "@/lib/test-store";
import { getSupabaseAdminClient } from "@/lib/supabase-server";

export type TestEvaluationsStorageMode = "memory" | "supabase";

export interface UpsertTestEvaluationInput {
  threadId: string;
  threadTitle: string;
  messageId: string;
  turnId?: string;
  verdict: TestVerdict;
  question: string;
  answer: string;
  agentId?: string;
  modelId?: string;
  comment?: string;
}

export interface ThreadTestStateResult {
  session: StoreTestSession | null;
  evaluations: StoreTestEvaluation[];
  evaluations_by_message_id: Record<string, StoreTestEvaluation>;
}

export interface TestEvaluationsRepository {
  ensureSessionForThread(threadId: string, title: string): Promise<StoreTestSession>;
  getThreadTestState(threadId: string): Promise<ThreadTestStateResult>;
  upsertEvaluation(input: UpsertTestEvaluationInput): Promise<{ session: StoreTestSession; evaluation: StoreTestEvaluation }>;
  finalizeSession(sessionId: string): Promise<StoreTestSession | null>;
  listSessions(): Promise<StoreTestSession[]>;
}

interface TestSessionRow {
  id: string;
  thread_id: string;
  thread_title: string;
  status: "active" | "completed";
  evaluated_count: number;
  correct_count: number;
  partial_count: number;
  incorrect_count: number;
  score_percent: number;
  started_at: string;
  ended_at: string | null;
  created_at: string;
  updated_at: string;
}

interface TestEvaluationRow {
  id: string;
  session_id: string;
  thread_id: string;
  message_id: string;
  turn_id: string | null;
  verdict: TestVerdict;
  score: number;
  question: string;
  answer: string;
  agent_id: string | null;
  model_id: string | null;
  comment: string | null;
  created_at: string;
  updated_at: string;
}

function getConfiguredStorageMode(): TestEvaluationsStorageMode {
  const rawMode = process.env.TEST_EVALUATIONS_STORAGE?.trim().toLowerCase();
  if (rawMode === "supabase") return "supabase";
  return "memory";
}

function mapSession(row: TestSessionRow): StoreTestSession {
  return {
    id: row.id,
    thread_id: row.thread_id,
    title: row.thread_title,
    status: row.status,
    created_at: row.created_at,
    updated_at: row.updated_at,
    started_at: row.started_at,
    ended_at: row.ended_at ?? undefined,
    metrics: {
      evaluated_count: row.evaluated_count,
      correct_count: row.correct_count,
      partial_count: row.partial_count,
      incorrect_count: row.incorrect_count,
      score_percent: row.score_percent,
    },
  };
}

function mapEvaluation(row: TestEvaluationRow): StoreTestEvaluation {
  return {
    id: row.id,
    session_id: row.session_id,
    thread_id: row.thread_id,
    message_id: row.message_id,
    turn_id: row.turn_id ?? undefined,
    verdict: row.verdict,
    score: row.score,
    question: row.question,
    answer: row.answer,
    agent_id: row.agent_id ?? undefined,
    model_id: row.model_id ?? undefined,
    comment: row.comment ?? undefined,
    created_at: row.created_at,
    updated_at: row.updated_at,
  };
}

function buildThreadState(session: StoreTestSession | null, evaluations: StoreTestEvaluation[]): ThreadTestStateResult {
  return {
    session,
    evaluations,
    evaluations_by_message_id: Object.fromEntries(evaluations.map((item) => [item.message_id, item])),
  };
}

async function recomputeAndPersistSessionMetrics(sessionId: string) {
  const supabase = getSupabaseAdminClient();
  const { data: evaluationRows, error: evaluationsError } = await supabase
    .from("test_evaluations")
    .select("verdict, score")
    .eq("session_id", sessionId);

  if (evaluationsError) {
    throw evaluationsError;
  }

  const rows = evaluationRows ?? [];
  const evaluatedCount = rows.length;
  const correctCount = rows.filter((item) => item.verdict === "correct").length;
  const partialCount = rows.filter((item) => item.verdict === "partial").length;
  const incorrectCount = rows.filter((item) => item.verdict === "incorrect").length;
  const totalPoints = rows.reduce((acc, item) => acc + Number(item.score ?? 0), 0);
  const scorePercent = evaluatedCount > 0 ? Math.round((totalPoints / evaluatedCount) * 100) : 0;

  const { data: updatedRow, error: updateError } = await supabase
    .from("test_sessions")
    .update({
      evaluated_count: evaluatedCount,
      correct_count: correctCount,
      partial_count: partialCount,
      incorrect_count: incorrectCount,
      score_percent: scorePercent,
    })
    .eq("id", sessionId)
    .select("*")
    .single<TestSessionRow>();

  if (updateError) {
    throw updateError;
  }

  return mapSession(updatedRow);
}

async function fetchActiveSessionForThread(threadId: string) {
  const supabase = getSupabaseAdminClient();
  const { data, error } = await supabase
    .from("test_sessions")
    .select("*")
    .eq("thread_id", threadId)
    .eq("status", "active")
    .order("updated_at", { ascending: false })
    .limit(1);

  if (error) {
    throw error;
  }

  const row = (data?.[0] as TestSessionRow | undefined) ?? null;
  return row ? mapSession(row) : null;
}

async function fetchLatestSessionForThread(threadId: string) {
  const supabase = getSupabaseAdminClient();
  const { data, error } = await supabase
    .from("test_sessions")
    .select("*")
    .eq("thread_id", threadId)
    .order("updated_at", { ascending: false })
    .limit(1);

  if (error) {
    throw error;
  }

  const row = (data?.[0] as TestSessionRow | undefined) ?? null;
  return row ? mapSession(row) : null;
}

const memoryRepository: TestEvaluationsRepository = {
  ensureSessionForThread: async (threadId: string, title: string) => ensureTestSessionForThread(threadId, title),
  getThreadTestState: async (threadId: string) => getInMemoryThreadTestState(threadId),
  upsertEvaluation: async (input: UpsertTestEvaluationInput) => upsertThreadEvaluation(input),
  finalizeSession: async (sessionId: string) => finalizeInMemoryTestSession(sessionId),
  listSessions: async () => listInMemoryTestSessions(),
};

const supabaseRepository: TestEvaluationsRepository = {
  async ensureSessionForThread(threadId: string, title: string) {
    const supabase = getSupabaseAdminClient();
    const activeSession = await fetchActiveSessionForThread(threadId);

    if (activeSession) {
      if (title && title !== activeSession.title) {
        const { data, error } = await supabase
          .from("test_sessions")
          .update({ thread_title: title })
          .eq("id", activeSession.id)
          .select("*")
          .single<TestSessionRow>();

        if (error) {
          throw error;
        }

        return mapSession(data);
      }

      return activeSession;
    }

    const { data, error } = await supabase
      .from("test_sessions")
      .insert({
        thread_id: threadId,
        thread_title: title,
        status: "active",
        source: "console",
      })
      .select("*")
      .single<TestSessionRow>();

    if (error) {
      throw error;
    }

    return mapSession(data);
  },

  async getThreadTestState(threadId: string) {
    const supabase = getSupabaseAdminClient();
    const session = (await fetchActiveSessionForThread(threadId)) ?? (await fetchLatestSessionForThread(threadId));
    if (!session) {
      return buildThreadState(null, []);
    }

    const { data, error } = await supabase
      .from("test_evaluations")
      .select("*")
      .eq("session_id", session.id)
      .order("created_at", { ascending: true });

    if (error) {
      throw error;
    }

    const evaluations = ((data ?? []) as TestEvaluationRow[]).map(mapEvaluation);
    return buildThreadState(session, evaluations);
  },

  async upsertEvaluation(input: UpsertTestEvaluationInput) {
    const supabase = getSupabaseAdminClient();
    const session = await this.ensureSessionForThread(input.threadId, input.threadTitle);

    const { data, error } = await supabase
      .from("test_evaluations")
      .upsert(
        {
          session_id: session.id,
          thread_id: input.threadId,
          message_id: input.messageId,
          turn_id: input.turnId ?? null,
          verdict: input.verdict,
          score: input.verdict === "correct" ? 1 : input.verdict === "partial" ? 0.5 : 0,
          question: input.question,
          answer: input.answer,
          agent_id: input.agentId ?? null,
          model_id: input.modelId ?? null,
          comment: input.comment ?? null,
        },
        { onConflict: "session_id,message_id" },
      )
      .select("*")
      .single<TestEvaluationRow>();

    if (error) {
      throw error;
    }

    const updatedSession = await recomputeAndPersistSessionMetrics(session.id);
    return {
      session: updatedSession,
      evaluation: mapEvaluation(data),
    };
  },

  async finalizeSession(sessionId: string) {
    const supabase = getSupabaseAdminClient();
    const recomputed = await recomputeAndPersistSessionMetrics(sessionId);

    const { data, error } = await supabase
      .from("test_sessions")
      .update({
        status: "completed",
        ended_at: new Date().toISOString(),
      })
      .eq("id", sessionId)
      .select("*")
      .single<TestSessionRow>();

    if (error) {
      throw error;
    }

    return mapSession({
      ...data,
      evaluated_count: recomputed.metrics.evaluated_count,
      correct_count: recomputed.metrics.correct_count,
      partial_count: recomputed.metrics.partial_count,
      incorrect_count: recomputed.metrics.incorrect_count,
      score_percent: recomputed.metrics.score_percent,
    });
  },

  async listSessions() {
    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("test_sessions")
      .select("*")
      .order("updated_at", { ascending: false });

    if (error) {
      throw error;
    }

    return ((data ?? []) as TestSessionRow[]).map(mapSession);
  },
};

export function getTestEvaluationsRepository(): TestEvaluationsRepository {
  const mode = getConfiguredStorageMode();
  if (mode === "supabase") {
    return supabaseRepository;
  }
  return memoryRepository;
}

export function getTestEvaluationsStorageMode() {
  return getConfiguredStorageMode();
}
