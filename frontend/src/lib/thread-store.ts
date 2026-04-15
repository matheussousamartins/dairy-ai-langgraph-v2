interface StoreMessage {
  id: string;
  type: "human" | "ai";
  content: string;
  created_at: string;
  response_metadata?: {
    model_name?: string;
  };
  tool_calls?: unknown[];
}

interface StoreThread {
  thread_id: string;
  created_at: string;
  values: {
    messages: StoreMessage[];
  };
}

interface ThreadStore {
  threads: Record<string, StoreThread>;
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

function getStore(): ThreadStore {
  if (!global.__dairyThreadStore__) {
    global.__dairyThreadStore__ = { threads: {} };
  }
  return global.__dairyThreadStore__;
}

export function createThread(): StoreThread {
  const store = getStore();
  const thread_id = genId("thread");
  const thread: StoreThread = {
    thread_id,
    created_at: nowIso(),
    values: { messages: [] },
  };
  store.threads[thread_id] = thread;
  return thread;
}

export function listThreads(): StoreThread[] {
  const store = getStore();
  return Object.values(store.threads).sort((a, b) => {
    return Date.parse(b.created_at) - Date.parse(a.created_at);
  });
}

export function getThread(threadId: string): StoreThread | null {
  const store = getStore();
  return store.threads[threadId] ?? null;
}

export function appendHumanMessage(threadId: string, content: string): StoreMessage | null {
  const thread = getThread(threadId);
  if (!thread) return null;
  const msg: StoreMessage = {
    id: genId("human"),
    type: "human",
    content,
    created_at: nowIso(),
  };
  thread.values.messages.push(msg);
  return msg;
}

export function appendAiMessage(
  threadId: string,
  content: string,
  modelName: string,
  toolCalls?: unknown[],
): StoreMessage | null {
  const thread = getThread(threadId);
  if (!thread) return null;
  const msg: StoreMessage = {
    id: genId("ai"),
    type: "ai",
    content,
    created_at: nowIso(),
    response_metadata: { model_name: modelName },
    tool_calls: toolCalls ?? [],
  };
  thread.values.messages.push(msg);
  return msg;
}

