import "server-only";

import { createHash } from "node:crypto";

import { getSupabaseAdminClient } from "@/lib/supabase-server";
import { summarizeThreadTitleFromMessages } from "@/lib/thread-title";

export interface StoreMessage {
  id: string;
  type: "human" | "ai";
  content: string;
  created_at: string;
  turn_id?: string;
  response_metadata?: {
    model_name?: string;
    agent_id?: string;
  };
  tool_calls?: unknown[];
  trace?: StoreTraceEvent[];
}

export interface StoreTraceEvent {
  type: "node_start" | "node_end" | "tool_call" | "tool_result";
  node?: string;
  tool?: string;
  input?: string;
  output?: string;
  ts: number;
}

export interface StoreTraceRecord {
  id: string;
  thread_id: string;
  turn_id: string;
  created_at: string;
  response_metadata?: {
    model_name?: string;
    agent_id?: string;
  };
  trace: StoreTraceEvent[];
}

export interface StoreThread {
  thread_id: string;
  owner_id: string;
  created_at: string;
  summary?: {
    title: string;
    preview: string;
    message_count: number;
    question: string;
    last_agent_id?: string;
  };
  values: {
    messages: StoreMessage[];
  };
}

export interface StoreThreadSummary {
  thread_id: string;
  created_at: string;
  summary: {
    title: string;
    preview: string;
    message_count: number;
    question: string;
    last_agent_id?: string;
  };
}

type ThreadsStorageMode = "memory" | "supabase";

interface ThreadStore {
  threads: Record<string, StoreThread>;
  traces: Record<string, StoreTraceRecord>;
}

interface ThreadRow {
  id: string;
  owner_id: string;
  title: string;
  preview: string;
  message_count: number;
  question: string;
  last_agent_id: string | null;
  created_at: string;
  updated_at: string;
}

interface ThreadMessageRow {
  id: string;
  thread_id: string;
  owner_id: string;
  type: "human" | "ai";
  content: string;
  created_at: string;
  turn_id: string | null;
  response_metadata: {
    model_name?: string;
    agent_id?: string;
  } | null;
  tool_calls: unknown[] | null;
}

interface ThreadTraceRow {
  id: string;
  thread_id: string;
  owner_id: string;
  turn_id: string;
  created_at: string;
  response_metadata: {
    model_name?: string;
    agent_id?: string;
  } | null;
  trace: StoreTraceEvent[] | null;
}

interface ThreadsRepository {
  createThread(ownerId: string): Promise<StoreThread>;
  listThreads(ownerId: string): Promise<StoreThreadSummary[]>;
  getThread(ownerId: string, threadId: string): Promise<StoreThread | null>;
  getThreadWithTraces(ownerId: string, threadId: string): Promise<StoreThread | null>;
  appendHumanMessage(ownerId: string, threadId: string, content: string, turnId?: string): Promise<StoreMessage | null>;
  appendAiMessage(
    ownerId: string,
    threadId: string,
    content: string,
    modelName: string,
    agentId?: string,
    toolCalls?: unknown[],
    turnId?: string,
  ): Promise<StoreMessage | null>;
  appendTrace(
    ownerId: string,
    threadId: string,
    turnId: string,
    trace: StoreTraceEvent[],
    modelName?: string,
    agentId?: string,
  ): Promise<StoreTraceRecord | null>;
}

declare global {
  // eslint-disable-next-line no-var
  var __dairyThreadStore__: ThreadStore | undefined;
}

function nowIso() {
  return new Date().toISOString();
}

function genId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function genTurnId() {
  return `turn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getStore(): ThreadStore {
  if (!global.__dairyThreadStore__) {
    global.__dairyThreadStore__ = { threads: {}, traces: {} };
  }
  return global.__dairyThreadStore__;
}

function normalizeStringArray(value: unknown) {
  return Array.isArray(value) ? value : [];
}

function buildThreadSummaryFromMessages(
  threadId: string,
  messages: StoreMessage[],
): NonNullable<StoreThread["summary"]> {
  const lastMessage = messages.at(-1)?.content ?? "";
  const lastAssistant = [...messages].reverse().find((message) => message.type === "ai");
  const firstRelevantUser =
    messages.find((message) => message.type === "human" && message.content.trim())?.content ?? "";
  const title = summarizeThreadTitleFromMessages(
    messages.map((message) => ({
      role: message.type === "human" ? "user" : "assistant",
      content: message.content,
    })),
    threadId,
  );

  return {
    title,
    preview: lastMessage,
    message_count: messages.length,
    question: firstRelevantUser,
    last_agent_id: lastAssistant?.response_metadata?.agent_id,
  };
}

function buildThreadSummary(thread: StoreThread): StoreThreadSummary {
  const summary = thread.summary ?? buildThreadSummaryFromMessages(thread.thread_id, thread.values.messages);
  return {
    thread_id: thread.thread_id,
    created_at: thread.created_at,
    summary,
  };
}

function deriveThreadOwnerId(token: string) {
  return createHash("sha256").update(token).digest("hex").slice(0, 32);
}

function getConfiguredStorageMode(): ThreadsStorageMode {
  const rawMode = process.env.THREADS_STORAGE?.trim().toLowerCase();
  if (rawMode === "supabase") return "supabase";
  if (process.env.TEST_EVALUATIONS_STORAGE?.trim().toLowerCase() === "supabase") {
    return "supabase";
  }
  return "memory";
}

function mapThreadRow(row: ThreadRow, messages: StoreMessage[] = []): StoreThread {
  return {
    thread_id: row.id,
    owner_id: row.owner_id,
    created_at: row.created_at,
    summary: {
      title: row.title,
      preview: row.preview,
      message_count: row.message_count,
      question: row.question,
      last_agent_id: row.last_agent_id ?? undefined,
    },
    values: {
      messages,
    },
  };
}

function mapMessageRow(row: ThreadMessageRow): StoreMessage {
  return {
    id: row.id,
    type: row.type,
    content: row.content,
    created_at: row.created_at,
    turn_id: row.turn_id ?? undefined,
    response_metadata: row.response_metadata ?? undefined,
    tool_calls: normalizeStringArray(row.tool_calls),
  };
}

function mapTraceRow(row: ThreadTraceRow): StoreTraceRecord {
  return {
    id: row.id,
    thread_id: row.thread_id,
    turn_id: row.turn_id,
    created_at: row.created_at,
    response_metadata: row.response_metadata ?? undefined,
    trace: Array.isArray(row.trace) ? row.trace : [],
  };
}

async function recomputeAndPersistThreadSummary(ownerId: string, threadId: string) {
  const supabase = getSupabaseAdminClient();
  const { data, error } = await supabase
    .from("console_thread_messages")
    .select("*")
    .eq("owner_id", ownerId)
    .eq("thread_id", threadId)
    .order("created_at", { ascending: true });

  if (error) {
    throw error;
  }

  const messages = ((data ?? []) as ThreadMessageRow[]).map(mapMessageRow);
  const summary = buildThreadSummaryFromMessages(threadId, messages);

  const { data: updatedRow, error: updateError } = await supabase
    .from("console_threads")
    .update({
      title: summary.title,
      preview: summary.preview,
      message_count: summary.message_count,
      question: summary.question,
      last_agent_id: summary.last_agent_id ?? null,
    })
    .eq("owner_id", ownerId)
    .eq("id", threadId)
    .select("*")
    .single<ThreadRow>();

  if (updateError) {
    throw updateError;
  }

  return mapThreadRow(updatedRow, messages);
}

const memoryRepository: ThreadsRepository = {
  async createThread(ownerId: string) {
    const store = getStore();
    const thread_id = genId("thread");
    const thread: StoreThread = {
      thread_id,
      owner_id: ownerId,
      created_at: nowIso(),
      summary: {
        title: "Nova sessão",
        preview: "",
        message_count: 0,
        question: "",
      },
      values: { messages: [] },
    };
    store.threads[thread_id] = thread;
    return thread;
  },

  async listThreads(ownerId: string) {
    const store = getStore();
    return Object.values(store.threads)
      .filter((thread) => thread.owner_id === ownerId)
      .sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
      .map(buildThreadSummary);
  },

  async getThread(ownerId: string, threadId: string) {
    const store = getStore();
    const thread = store.threads[threadId] ?? null;
    if (!thread || thread.owner_id !== ownerId) return null;
    return thread;
  },

  async getThreadWithTraces(ownerId: string, threadId: string) {
    const thread = await this.getThread(ownerId, threadId);
    if (!thread) return null;
    const store = getStore();
    const traceByTurnId = new Map(
      Object.values(store.traces)
        .filter((trace) => trace.thread_id === threadId)
        .map((trace) => [trace.turn_id, trace]),
    );

    return {
      ...thread,
      values: {
        messages: thread.values.messages.map((message) => {
          const traceRecord = message.turn_id ? traceByTurnId.get(message.turn_id) : undefined;
          return traceRecord ? { ...message, trace: traceRecord.trace } : message;
        }),
      },
    };
  },

  async appendHumanMessage(ownerId: string, threadId: string, content: string, turnId?: string) {
    const thread = await this.getThread(ownerId, threadId);
    if (!thread) return null;
    const message: StoreMessage = {
      id: genId("human"),
      type: "human",
      content,
      created_at: nowIso(),
      turn_id: turnId ?? genTurnId(),
    };
    thread.values.messages.push(message);
    thread.summary = buildThreadSummaryFromMessages(threadId, thread.values.messages);
    return message;
  },

  async appendAiMessage(ownerId: string, threadId: string, content: string, modelName: string, agentId?: string, toolCalls?: unknown[], turnId?: string) {
    const thread = await this.getThread(ownerId, threadId);
    if (!thread) return null;
    const message: StoreMessage = {
      id: genId("ai"),
      type: "ai",
      content,
      created_at: nowIso(),
      turn_id: turnId ?? genTurnId(),
      response_metadata: {
        model_name: modelName,
        agent_id: agentId,
      },
      tool_calls: toolCalls ?? [],
    };
    thread.values.messages.push(message);
    thread.summary = buildThreadSummaryFromMessages(threadId, thread.values.messages);
    return message;
  },

  async appendTrace(ownerId: string, threadId: string, turnId: string, trace: StoreTraceEvent[], modelName?: string, agentId?: string) {
    const thread = await this.getThread(ownerId, threadId);
    if (!thread || trace.length === 0) return null;
    const store = getStore();
    const record: StoreTraceRecord = {
      id: genId("trace"),
      thread_id: threadId,
      turn_id: turnId,
      created_at: nowIso(),
      response_metadata: {
        model_name: modelName,
        agent_id: agentId,
      },
      trace,
    };
    store.traces[record.id] = record;
    return record;
  },
};

const supabaseRepository: ThreadsRepository = {
  async createThread(ownerId: string) {
    const supabase = getSupabaseAdminClient();
    const row = {
      id: genId("thread"),
      owner_id: ownerId,
      title: "Nova sessão",
      preview: "",
      message_count: 0,
      question: "",
      last_agent_id: null,
    };
    const { data, error } = await supabase
      .from("console_threads")
      .insert(row)
      .select("*")
      .single<ThreadRow>();

    if (error) {
      throw error;
    }

    return mapThreadRow(data, []);
  },

  async listThreads(ownerId: string) {
    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("console_threads")
      .select("*")
      .eq("owner_id", ownerId)
      .order("updated_at", { ascending: false });

    if (error) {
      throw error;
    }

    return ((data ?? []) as ThreadRow[]).map((row) => ({
      thread_id: row.id,
      created_at: row.created_at,
      summary: {
        title: row.title,
        preview: row.preview,
        message_count: row.message_count,
        question: row.question,
        last_agent_id: row.last_agent_id ?? undefined,
      },
    }));
  },

  async getThread(ownerId: string, threadId: string) {
    const supabase = getSupabaseAdminClient();
    const { data: threadRow, error: threadError } = await supabase
      .from("console_threads")
      .select("*")
      .eq("owner_id", ownerId)
      .eq("id", threadId)
      .maybeSingle<ThreadRow>();

    if (threadError) {
      throw threadError;
    }

    if (!threadRow) {
      return null;
    }

    const { data: messageRows, error: messagesError } = await supabase
      .from("console_thread_messages")
      .select("*")
      .eq("owner_id", ownerId)
      .eq("thread_id", threadId)
      .order("created_at", { ascending: true });

    if (messagesError) {
      throw messagesError;
    }

    const messages = ((messageRows ?? []) as ThreadMessageRow[]).map(mapMessageRow);
    return mapThreadRow(threadRow, messages);
  },

  async getThreadWithTraces(ownerId: string, threadId: string) {
    const thread = await this.getThread(ownerId, threadId);
    if (!thread) return null;

    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("console_thread_traces")
      .select("*")
      .eq("owner_id", ownerId)
      .eq("thread_id", threadId)
      .order("created_at", { ascending: true });

    if (error) {
      throw error;
    }

    const traceByTurnId = new Map(
      ((data ?? []) as ThreadTraceRow[]).map((row) => {
        const trace = mapTraceRow(row);
        return [trace.turn_id, trace] as const;
      }),
    );

    return {
      ...thread,
      values: {
        messages: thread.values.messages.map((message) => {
          const traceRecord = message.turn_id ? traceByTurnId.get(message.turn_id) : undefined;
          return traceRecord ? { ...message, trace: traceRecord.trace } : message;
        }),
      },
    };
  },

  async appendHumanMessage(ownerId: string, threadId: string, content: string, turnId?: string) {
    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("console_thread_messages")
      .insert({
        id: genId("human"),
        thread_id: threadId,
        owner_id: ownerId,
        type: "human",
        content,
        turn_id: turnId ?? genTurnId(),
        response_metadata: {},
        tool_calls: [],
      })
      .select("*")
      .single<ThreadMessageRow>();

    if (error) {
      throw error;
    }

    await recomputeAndPersistThreadSummary(ownerId, threadId);
    return mapMessageRow(data);
  },

  async appendAiMessage(ownerId: string, threadId: string, content: string, modelName: string, agentId?: string, toolCalls?: unknown[], turnId?: string) {
    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("console_thread_messages")
      .insert({
        id: genId("ai"),
        thread_id: threadId,
        owner_id: ownerId,
        type: "ai",
        content,
        turn_id: turnId ?? genTurnId(),
        response_metadata: {
          model_name: modelName,
          agent_id: agentId,
        },
        tool_calls: toolCalls ?? [],
      })
      .select("*")
      .single<ThreadMessageRow>();

    if (error) {
      throw error;
    }

    await recomputeAndPersistThreadSummary(ownerId, threadId);
    return mapMessageRow(data);
  },

  async appendTrace(ownerId: string, threadId: string, turnId: string, trace: StoreTraceEvent[], modelName?: string, agentId?: string) {
    if (trace.length === 0) return null;

    const supabase = getSupabaseAdminClient();
    const { data, error } = await supabase
      .from("console_thread_traces")
      .insert({
        id: genId("trace"),
        thread_id: threadId,
        owner_id: ownerId,
        turn_id: turnId,
        response_metadata: {
          model_name: modelName,
          agent_id: agentId,
        },
        trace,
      })
      .select("*")
      .single<ThreadTraceRow>();

    if (error) {
      throw error;
    }

    return mapTraceRow(data);
  },
};

function getThreadsRepository() {
  return getConfiguredStorageMode() === "supabase" ? supabaseRepository : memoryRepository;
}

export function deriveConsoleThreadOwnerId(token: string) {
  return deriveThreadOwnerId(token);
}

export async function createThread(ownerId: string) {
  return getThreadsRepository().createThread(ownerId);
}

export async function listThreads(ownerId: string) {
  return getThreadsRepository().listThreads(ownerId);
}

export async function getThread(ownerId: string, threadId: string) {
  return getThreadsRepository().getThread(ownerId, threadId);
}

export async function getThreadWithTraces(ownerId: string, threadId: string) {
  return getThreadsRepository().getThreadWithTraces(ownerId, threadId);
}

export async function appendHumanMessage(ownerId: string, threadId: string, content: string, turnId?: string) {
  return getThreadsRepository().appendHumanMessage(ownerId, threadId, content, turnId);
}

export async function appendAiMessage(
  ownerId: string,
  threadId: string,
  content: string,
  modelName: string,
  agentId?: string,
  toolCalls?: unknown[],
  turnId?: string,
) {
  return getThreadsRepository().appendAiMessage(ownerId, threadId, content, modelName, agentId, toolCalls, turnId);
}

export async function appendTrace(
  ownerId: string,
  threadId: string,
  turnId: string,
  trace: StoreTraceEvent[],
  modelName?: string,
  agentId?: string,
) {
  return getThreadsRepository().appendTrace(ownerId, threadId, turnId, trace, modelName, agentId);
}
