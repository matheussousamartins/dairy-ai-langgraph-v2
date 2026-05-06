export type TestVerdict = "correct" | "partial" | "incorrect";
export type EvaluationErrorCategory =
  | "retrieval"
  | "routing"
  | "consolidation"
  | "hallucination"
  | "missing_kb"
  | "regulatory_conflict"
  | "wrong_scope"
  | "ui"
  | "other";

export type EvaluationStatus = "new" | "accepted" | "triaged" | "fixed" | "ignored" | "regression_test_added";
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export interface StoreTestEvaluation {
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

export interface StoreTestSessionMetrics {
  evaluated_count: number;
  correct_count: number;
  partial_count: number;
  incorrect_count: number;
  score_percent: number;
}

export interface StoreTestSession {
  id: string;
  thread_id: string;
  title: string;
  status: "active" | "completed";
  created_at: string;
  updated_at: string;
  started_at: string;
  ended_at?: string;
  metrics: StoreTestSessionMetrics;
}

interface TestStore {
  sessions: Record<string, StoreTestSession>;
  evaluations: Record<string, StoreTestEvaluation>;
}

declare global {
  // eslint-disable-next-line no-var
  var __dairyTestStore__: TestStore | undefined;
}

function nowIso() {
  return new Date().toISOString();
}

function genId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getStore(): TestStore {
  if (!global.__dairyTestStore__) {
    global.__dairyTestStore__ = { sessions: {}, evaluations: {} };
  }
  return global.__dairyTestStore__;
}

function getVerdictScore(verdict: TestVerdict) {
  if (verdict === "correct") return 1;
  if (verdict === "partial") return 0.5;
  return 0;
}

function emptyMetrics(): StoreTestSessionMetrics {
  return {
    evaluated_count: 0,
    correct_count: 0,
    partial_count: 0,
    incorrect_count: 0,
    score_percent: 0,
  };
}

function recomputeSessionMetrics(sessionId: string) {
  const store = getStore();
  const evaluations = Object.values(store.evaluations).filter((item) => item.session_id === sessionId);
  const metrics = emptyMetrics();

  evaluations.forEach((item) => {
    metrics.evaluated_count += 1;
    if (item.verdict === "correct") metrics.correct_count += 1;
    if (item.verdict === "partial") metrics.partial_count += 1;
    if (item.verdict === "incorrect") metrics.incorrect_count += 1;
  });

  const totalPoints = evaluations.reduce((acc, item) => acc + item.score, 0);
  metrics.score_percent = metrics.evaluated_count > 0 ? Math.round((totalPoints / metrics.evaluated_count) * 100) : 0;
  return metrics;
}

function updateSessionMetrics(sessionId: string) {
  const store = getStore();
  const session = store.sessions[sessionId];
  if (!session) return null;
  const metrics = recomputeSessionMetrics(sessionId);
  const updated = {
    ...session,
    metrics,
    updated_at: nowIso(),
  };
  store.sessions[sessionId] = updated;
  return updated;
}

function getLatestSessionForThread(threadId: string) {
  const store = getStore();
  return Object.values(store.sessions)
    .filter((session) => session.thread_id === threadId)
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))[0] ?? null;
}

function getActiveSessionForThread(threadId: string) {
  const store = getStore();
  return Object.values(store.sessions)
    .filter((session) => session.thread_id === threadId && session.status === "active")
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))[0] ?? null;
}

export function ensureTestSessionForThread(threadId: string, title: string) {
  const store = getStore();
  const existing = getActiveSessionForThread(threadId);
  if (existing) {
    const updated = { ...existing, title: title || existing.title, updated_at: nowIso() };
    store.sessions[updated.id] = updated;
    return updated;
  }

  const now = nowIso();
  const session: StoreTestSession = {
    id: genId("test-session"),
    thread_id: threadId,
    title,
    status: "active",
    created_at: now,
    updated_at: now,
    started_at: now,
    metrics: emptyMetrics(),
  };
  store.sessions[session.id] = session;
  return session;
}

export function upsertThreadEvaluation(input: {
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
  environment?: string;
  appVersion?: string;
  gitSha?: string;
  ragArchitecture?: string;
  promptVersion?: string;
  retrievalConfigVersion?: string;
  errorCategory?: EvaluationErrorCategory;
  expectedAnswer?: string;
  status?: EvaluationStatus;
  answerSource?: string;
  chosenAgentIds?: number[];
  primaryAgentId?: string;
  topRagScore?: number;
  ragSources?: string[];
  ragSearchCount?: number;
  nodeCount?: number;
  latencyMs?: number;
  webFallbackUsed?: boolean;
  metadata?: Record<string, JsonValue>;
}) {
  const store = getStore();
  const session = ensureTestSessionForThread(input.threadId, input.threadTitle);
  const existing = Object.values(store.evaluations).find(
    (item) => item.session_id === session.id && item.message_id === input.messageId,
  );

  const now = nowIso();
  const record: StoreTestEvaluation = {
    id: existing?.id ?? genId("test-eval"),
    session_id: session.id,
    thread_id: input.threadId,
    message_id: input.messageId,
    turn_id: input.turnId,
    verdict: input.verdict,
    score: getVerdictScore(input.verdict),
    question: input.question,
    answer: input.answer,
    agent_id: input.agentId,
    model_id: input.modelId,
    comment: input.comment,
    evaluator_id: input.evaluatorId,
    environment: input.environment,
    app_version: input.appVersion,
    git_sha: input.gitSha,
    rag_architecture: input.ragArchitecture,
    prompt_version: input.promptVersion,
    retrieval_config_version: input.retrievalConfigVersion,
    error_category: input.verdict === "correct" ? undefined : input.errorCategory,
    expected_answer: input.expectedAnswer,
    status: input.status ?? (input.verdict === "correct" ? "accepted" : "new"),
    answer_source: input.answerSource,
    chosen_agent_ids: input.chosenAgentIds,
    primary_agent_id: input.primaryAgentId,
    top_rag_score: input.topRagScore,
    rag_sources: input.ragSources,
    rag_search_count: input.ragSearchCount,
    node_count: input.nodeCount,
    latency_ms: input.latencyMs,
    web_fallback_used: input.webFallbackUsed,
    metadata: input.metadata,
    created_at: existing?.created_at ?? now,
    updated_at: now,
  };

  store.evaluations[record.id] = record;
  const updatedSession = updateSessionMetrics(session.id);
  return { session: updatedSession ?? session, evaluation: record };
}

export function finalizeTestSession(sessionId: string) {
  const store = getStore();
  const session = store.sessions[sessionId];
  if (!session) return null;
  const now = nowIso();
  const updated: StoreTestSession = {
    ...session,
    status: "completed",
    updated_at: now,
    ended_at: now,
    metrics: recomputeSessionMetrics(sessionId),
  };
  store.sessions[sessionId] = updated;
  return updated;
}

export function getThreadTestState(threadId: string) {
  const session = getActiveSessionForThread(threadId) ?? getLatestSessionForThread(threadId);
  if (!session) {
    return {
      session: null,
      evaluations: [],
      evaluations_by_message_id: {} as Record<string, StoreTestEvaluation>,
    };
  }

  const store = getStore();
  const evaluations = Object.values(store.evaluations)
    .filter((item) => item.session_id === session.id)
    .sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at));

  const evaluationsByMessageId = Object.fromEntries(evaluations.map((item) => [item.message_id, item]));
  return {
    session,
    evaluations,
    evaluations_by_message_id: evaluationsByMessageId,
  };
}

export function listTestSessions() {
  const store = getStore();
  return Object.values(store.sessions).sort(
    (left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at),
  );
}
