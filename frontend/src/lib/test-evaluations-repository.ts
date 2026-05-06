import {
  ensureTestSessionForThread,
  finalizeTestSession as finalizeInMemoryTestSession,
  getThreadTestState as getInMemoryThreadTestState,
  listTestSessions as listInMemoryTestSessions,
  type EvaluationErrorCategory,
  type EvaluationStatus,
  type JsonValue,
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
  evaluatorId?: string;
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
  evaluator_id: string | null;
  environment: string | null;
  app_version: string | null;
  git_sha: string | null;
  rag_architecture: string | null;
  prompt_version: string | null;
  retrieval_config_version: string | null;
  error_category: EvaluationErrorCategory | null;
  expected_answer: string | null;
  status: EvaluationStatus | null;
  answer_source: string | null;
  chosen_agent_ids: number[] | null;
  primary_agent_id: string | null;
  top_rag_score: number | null;
  rag_sources: string[] | null;
  rag_search_count: number | null;
  node_count: number | null;
  latency_ms: number | null;
  web_fallback_used: boolean | null;
  metadata: Record<string, JsonValue> | null;
  created_at: string;
  updated_at: string;
}

function getConfiguredStorageMode(): TestEvaluationsStorageMode {
  const rawMode = process.env.TEST_EVALUATIONS_STORAGE?.trim().toLowerCase();
  if (rawMode === "supabase") return "supabase";
  return "memory";
}

function getRuntimeEnvironment() {
  return (
    process.env.RAILWAY_ENVIRONMENT_NAME ??
    process.env.VERCEL_ENV ??
    process.env.NODE_ENV ??
    "development"
  );
}

function getRuntimeAppVersion() {
  return process.env.NEXT_PUBLIC_APP_VERSION ?? process.env.APP_VERSION ?? null;
}

function getRuntimeGitSha() {
  return (
    process.env.RAILWAY_GIT_COMMIT_SHA ??
    process.env.VERCEL_GIT_COMMIT_SHA ??
    process.env.GIT_SHA ??
    null
  );
}

function getRuntimeRagArchitecture() {
  return process.env.RAG_ARCHITECTURE ?? process.env.NEXT_PUBLIC_RAG_ARCHITECTURE ?? null;
}

function normalizeMetadata(input?: Record<string, JsonValue>) {
  return {
    ...(input ?? {}),
    evaluation_schema_version: 2,
    evaluation_client: "console",
    captured_at: new Date().toISOString(),
  } satisfies Record<string, JsonValue>;
}

function scoreFromVerdict(verdict: TestVerdict) {
  if (verdict === "correct") return 1;
  if (verdict === "partial") return 0.5;
  return 0;
}

function statusFromVerdict(verdict: TestVerdict): EvaluationStatus {
  return verdict === "correct" ? "accepted" : "new";
}

function isSchemaCacheMissingColumnError(error: { code?: string; message?: string } | null) {
  if (!error) return false;
  const message = `${error.code ?? ""} ${error.message ?? ""}`.toLowerCase();
  return (
    message.includes("pgrst204") ||
    message.includes("schema cache") ||
    message.includes("could not find") ||
    message.includes("column") && message.includes("test_evaluations")
  );
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
    evaluator_id: row.evaluator_id ?? undefined,
    environment: row.environment ?? undefined,
    app_version: row.app_version ?? undefined,
    git_sha: row.git_sha ?? undefined,
    rag_architecture: row.rag_architecture ?? undefined,
    prompt_version: row.prompt_version ?? undefined,
    retrieval_config_version: row.retrieval_config_version ?? undefined,
    error_category: row.error_category ?? undefined,
    expected_answer: row.expected_answer ?? undefined,
    status: row.status ?? undefined,
    answer_source: row.answer_source ?? undefined,
    chosen_agent_ids: Array.isArray(row.chosen_agent_ids) ? row.chosen_agent_ids : undefined,
    primary_agent_id: row.primary_agent_id ?? undefined,
    top_rag_score: row.top_rag_score ?? undefined,
    rag_sources: Array.isArray(row.rag_sources) ? row.rag_sources : undefined,
    rag_search_count: row.rag_search_count ?? undefined,
    node_count: row.node_count ?? undefined,
    latency_ms: row.latency_ms ?? undefined,
    web_fallback_used: row.web_fallback_used ?? undefined,
    metadata: row.metadata ?? undefined,
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

function buildEvaluationPayload(
  sessionId: string,
  input: UpsertTestEvaluationInput,
  includeQualityColumns: boolean,
) {
  const metadata = normalizeMetadata(input.metadata);
  const runtimeEnvironment = getRuntimeEnvironment();
  const runtimeAppVersion = getRuntimeAppVersion();
  const runtimeGitSha = getRuntimeGitSha();
  const runtimeRagArchitecture = getRuntimeRagArchitecture();

  const basePayload = {
    session_id: sessionId,
    thread_id: input.threadId,
    message_id: input.messageId,
    turn_id: input.turnId ?? null,
    verdict: input.verdict,
    score: scoreFromVerdict(input.verdict),
    question: input.question,
    answer: input.answer,
    agent_id: input.agentId ?? null,
    model_id: input.modelId ?? null,
    comment: input.comment ?? null,
    metadata,
  };

  if (!includeQualityColumns) {
    return basePayload;
  }

  return {
    ...basePayload,
    evaluator_id: input.evaluatorId ?? null,
    environment: runtimeEnvironment,
    app_version: runtimeAppVersion,
    git_sha: runtimeGitSha,
    rag_architecture: runtimeRagArchitecture,
    prompt_version: input.metadata?.prompt_version?.toString() ?? null,
    retrieval_config_version: input.metadata?.retrieval_config_version?.toString() ?? null,
    error_category: input.verdict === "correct" ? null : input.errorCategory ?? null,
    expected_answer: input.expectedAnswer ?? null,
    status: statusFromVerdict(input.verdict),
    answer_source: input.answerSource ?? null,
    chosen_agent_ids: input.chosenAgentIds ?? [],
    primary_agent_id: input.primaryAgentId ?? input.agentId ?? null,
    top_rag_score: input.topRagScore ?? null,
    rag_sources: input.ragSources ?? [],
    rag_search_count: input.ragSearchCount ?? 0,
    node_count: input.nodeCount ?? 0,
    latency_ms: input.latencyMs ?? null,
    web_fallback_used: input.webFallbackUsed ?? false,
  };
}

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
        buildEvaluationPayload(session.id, input, true),
        { onConflict: "session_id,message_id" },
      )
      .select("*")
      .single<TestEvaluationRow>();

    if (error) {
      if (!isSchemaCacheMissingColumnError(error)) {
        throw error;
      }

      const retry = await supabase
        .from("test_evaluations")
        .upsert(
          buildEvaluationPayload(session.id, input, false),
          { onConflict: "session_id,message_id" },
        )
        .select("*")
        .single<TestEvaluationRow>();

      if (retry.error) {
        throw retry.error;
      }

      const updatedSession = await recomputeAndPersistSessionMetrics(session.id);
      return {
        session: updatedSession,
        evaluation: mapEvaluation(retry.data),
      };
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
