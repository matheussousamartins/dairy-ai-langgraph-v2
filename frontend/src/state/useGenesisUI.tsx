"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "./useAuth";

export type Role = "user" | "assistant";

export interface TraceEvent {
  type: "node_start" | "node_end" | "tool_call" | "tool_result";
  node?: string;
  tool?: string;
  input?: string;
  output?: string;
  ts: number;
}

export interface GenesisMessage {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  modelId?: string;
  usedTavily?: boolean;
  trace?: TraceEvent[];
}

interface RawMessage {
  id?: string;
  type?: string;
  content?: unknown;
  response_metadata?: {
    model_name?: string;
  };
  tool_calls?: unknown[];
  created_at?: string;
  additional_kwargs?: {
    created_at?: string;
  };
}

interface RawThread {
  thread_id: string;
  created_at?: string;
  values?: {
    messages?: RawMessage[];
  };
}

interface ModelsResponse {
  models?: Array<{
    id: string;
    label: string;
    input_cost?: number;
    output_cost?: number;
  }>;
}

interface StreamEventPayload {
  event?: string;
  text?: string;
  messages?: RawMessage[];
  detail?: string;
  // trace fields
  type?: string;
  node?: string;
  tool?: string;
  input?: string;
  output?: string;
  ts?: number;
}

function serializeContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : JSON.stringify(part)))
      .join("\n");
  }
  if (content && typeof content === "object") {
    try {
      return JSON.stringify(content);
    } catch {
      return String(content);
    }
  }
  return typeof content === "undefined" ? "" : String(content);
}

function normalizeMessage(message: RawMessage, index: number): GenesisMessage | null {
  if (message.type === "tool" || message.type === "function") {
    return null;
  }

  const content = serializeContent(message.content);
  if (content.includes('"type":"settings"')) {
    return null;
  }

  if (!content.trim()) {
    return null;
  }
  const role: Role = message.type === "human" ? "user" : "assistant";
  const modelId = message.response_metadata?.model_name;
  
  // Tenta extrair timestamp da mensagem
  let timestamp = Date.now();
  if (message.created_at) {
    const parsed = Date.parse(message.created_at);
    if (!isNaN(parsed)) timestamp = parsed;
  } else if (message.additional_kwargs?.created_at) {
    const parsed = Date.parse(message.additional_kwargs.created_at);
    if (!isNaN(parsed)) timestamp = parsed;
  } else if (message.id) {
    // Tenta extrair timestamp do ID se for um UUID v7 ou contiver timestamp
    const uuidMatch = message.id.match(/^[0-9a-f]{8}-/);
    if (uuidMatch) {
      // UUID v7 tem timestamp nos primeiros bytes, mas é complexo extrair
      // Mantém Date.now() como fallback
    }
  }
  
  return {
    id: message.id ?? `msg-${index}`,
    role,
    content,
    timestamp,
    modelId,
    usedTavily: Boolean(message.tool_calls && (message.tool_calls as unknown[]).length > 0),
  };
}

function normalizeThread(thread: RawThread) {
  const messages = Array.isArray(thread?.values?.messages) ? thread.values.messages : [];
  const threadCreatedAt = thread.created_at ? Date.parse(thread.created_at) : Date.now();
  
  // Normaliza mensagens e ajusta timestamps relativos se necessário
  const normalized = messages
    .map((m, index) => {
      const msg = normalizeMessage(m, index);
      if (!msg) return null;
      
      // Se a mensagem não tem timestamp real, usa timestamp incremental baseado na ordem
      if (!m.created_at && !m.additional_kwargs?.created_at) {
        // Adiciona 1 segundo por mensagem a partir da criação da thread
        msg.timestamp = threadCreatedAt + (index * 1000);
      }
      
      return msg;
    })
    .filter((msg): msg is GenesisMessage => Boolean(msg));
    
  const firstUser = normalized.find((msg) => msg.role === "user");
  const title = firstUser ? firstUser.content.slice(0, 42) : `Thread ${thread.thread_id.slice(0, 8)}`;
  const session: GenesisSession = {
    id: thread.thread_id,
    title,
    createdAt: threadCreatedAt,
  };
  return { session, messages: normalized };
}

export interface GenesisSession {
  id: string;
  title: string;
  createdAt: number;
}

export interface ModelOption {
  id: string;
  label: string;
  inputCost: number;
  outputCost: number;
}

interface GenesisUIState {
  isLoading: boolean;
  isSending: boolean;
  models: ModelOption[];
  selectedModelId: string;
  setSelectedModelId: (id: string) => void;
  useTavily: boolean;
  setUseTavily: (value: boolean) => void;
  sessions: GenesisSession[];
  currentSessionId: string;
  createSession: () => Promise<string | undefined>;
  selectSession: (id: string) => Promise<void>;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;
  messagesBySession: Record<string, GenesisMessage[]>;
  sendMessage: (content: string) => Promise<void>;
}

const GenesisUIContext = createContext<GenesisUIState | null>(null);

export function GenesisUIProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [useTavily, setUseTavily] = useState<boolean>(true);
  const [sessions, setSessions] = useState<GenesisSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [messagesBySession, setMessagesBySession] = useState<Record<string, GenesisMessage[]>>({});
  const { token, isReady: authReady, logout } = useAuth();
  const selectedModelRef = useRef<string>("");
  const useTavilyRef = useRef<boolean>(useTavily);
  const lastRunSettingsRef = useRef<Record<string, { modelId: string; useTavily: boolean }>>({});

  useEffect(() => {
    selectedModelRef.current = selectedModelId;
  }, [selectedModelId]);

  useEffect(() => {
    useTavilyRef.current = useTavily;
  }, [useTavily]);

  const loadModels = useCallback(async () => {
    if (!token) return;
    const res = await fetch("/api/models", {
      cache: "no-store",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status === 401) {
      logout();
      return;
    }
    if (!res.ok) return;
    const data = (await res.json()) as ModelsResponse;
    const mapped: ModelOption[] = (data.models ?? []).map((model) => ({
      id: model.id,
      label: model.label,
      inputCost: model.input_cost ?? 0,
      outputCost: model.output_cost ?? 0,
    }));
    setModels(mapped);
    setSelectedModelId((prev) => (prev ? prev : mapped[0]?.id ?? ""));
  }, [token, logout]);

  const annotateWithLastRun = (threadId: string, messages: GenesisMessage[]) => {
    const last = lastRunSettingsRef.current[threadId];
    if (!last) return messages;
    let lastAssistantIndex = -1;
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]?.role === "assistant") {
        lastAssistantIndex = i;
        break;
      }
    }
    if (lastAssistantIndex === -1) return messages;
    const cloned = [...messages];
    cloned[lastAssistantIndex] = {
      ...cloned[lastAssistantIndex],
      modelId: last.modelId,
      usedTavily: last.useTavily,
    };
    return cloned;
  };

  const fetchThread = useCallback(
    async (threadId: string) => {
      if (!token) return;
      const res = await fetch(`/api/threads/${threadId}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: "no-store",
      });
      if (res.status === 401) {
        logout();
        return;
      }
      if (!res.ok) return;
      const data = await res.json();
      const thread = data.thread;
      if (!thread) return;
      const { messages, session } = normalizeThread(thread);
      setMessagesBySession((prev) => ({ ...prev, [threadId]: annotateWithLastRun(threadId, messages) }));
      setSessions((prev) => {
        const exists = prev.find((s) => s.id === threadId);
        if (exists) {
          return prev.map((s) =>
            s.id === threadId ? { ...s, title: session.title, createdAt: session.createdAt } : s,
          );
        }
        return [session, ...prev];
      });
    },
    [token, logout],
  );

  const loadThreads = useCallback(async () => {
    if (!token) {
      return;
    }
    const res = await fetch("/api/threads", {
      method: "GET",
      cache: "no-store",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status === 401) {
      logout();
      return;
    }
    if (!res.ok) return;
    const data = (await res.json()) as { threads?: RawThread[] };
    const threads = Array.isArray(data.threads) ? data.threads : [];
    const nextSessions: GenesisSession[] = [];
    const nextMessages: Record<string, GenesisMessage[]> = {};
    threads.forEach((thread) => {
      const { session, messages } = normalizeThread(thread);
      nextSessions.push(session);
      nextMessages[session.id] = messages;
    });
    setSessions(nextSessions);
    setMessagesBySession(nextMessages);
    setCurrentSessionId((prev) => {
      if (prev && nextSessions.some((session) => session.id === prev)) {
        return prev;
      }
      return "";
    });

    const threadIds = nextSessions.map((session) => session.id);
    const batchSize = 5;
    for (let i = 0; i < threadIds.length; i += batchSize) {
      const chunk = threadIds.slice(i, i + batchSize);
      await Promise.all(
        chunk.map((id) =>
          fetchThread(id).catch((error) => {
            console.error("Falha ao hidratar thread", id, error);
          }),
        ),
      );
    }
  }, [token, logout, fetchThread]);

  useEffect(() => {
    if (!authReady) {
      return;
    }
    if (!token) {
      setIsLoading(false);
      setSessions([]);
      setMessagesBySession({});
      setCurrentSessionId("");
      return;
    }
    let cancelled = false;
    async function bootstrap() {
      setIsLoading(true);
      try {
        await Promise.all([loadModels(), loadThreads()]);
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }
    bootstrap().catch(console.error);
    return () => {
      cancelled = true;
    };
  }, [authReady, token, loadModels, loadThreads]);

  const createSession = useCallback(async (): Promise<string | undefined> => {
    if (!token) return;
    const res = await fetch("/api/threads", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
    });
    if (res.status === 401) {
      logout();
      return;
    }
    if (!res.ok) return;
    const data = (await res.json()) as { thread?: RawThread };
    const thread = data.thread;
    if (!thread) return;
    const { session } = normalizeThread(thread);
    setSessions((prev) => [session, ...prev]);
    setMessagesBySession((prev) => ({ ...prev, [session.id]: [] }));
    setCurrentSessionId(session.id);
    return session.id;
  }, [token, logout]);

  const selectSession = useCallback(async (id: string) => {
    setCurrentSessionId(id);
    await fetchThread(id);
  }, [fetchThread]);

  const renameSession = useCallback((id: string, title: string) => {
    setSessions((prev) => prev.map((session) => (session.id === id ? { ...session, title } : session)));
  }, []);

  const deleteSession = useCallback((id: string) => {
    setSessions((prev) => prev.filter((session) => session.id !== id));
    setCurrentSessionId((prevId) => (prevId === id ? "" : prevId));
    setMessagesBySession((prev) => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!token) return;
      const threadId = currentSessionId;
      if (!threadId) return;

      const activeModelId = selectedModelRef.current || selectedModelId || models[0]?.id || "";
      if (!activeModelId) {
        console.error("Nenhum modelo selecionado para envio da mensagem.");
        return;
      }
      const activeUseTavily = useTavilyRef.current;
      lastRunSettingsRef.current[threadId] = { modelId: activeModelId, useTavily: activeUseTavily };

      const optimistic: GenesisMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content,
        timestamp: Date.now(),
        modelId: activeModelId,
        usedTavily: activeUseTavily,
      };

      setMessagesBySession((prev) => {
        const existing = prev[threadId] ?? [];
        return { ...prev, [threadId]: [...existing, optimistic] };
      });
      setCurrentSessionId(threadId);

      setIsSending(true);

      const targetThreadId = threadId;
      const thinkingMessageId = `thinking-${Date.now()}`;
      const thinkingMessage: GenesisMessage = {
        id: thinkingMessageId,
        role: "assistant",
        content: "Pensando...",
        timestamp: Date.now(),
      };

      setMessagesBySession((prev) => {
        const existing = prev[targetThreadId] ?? [];
        return { ...prev, [targetThreadId]: [...existing, thinkingMessage] };
      });

      const removeThinkingMessage = () => {
        setMessagesBySession((prev) => {
          const existing = prev[targetThreadId] ?? [];
          return { ...prev, [targetThreadId]: existing.filter((msg) => msg.id !== thinkingMessageId) };
        });
      };

      let streamMessageId: string | null = null;
      const removeStreamMessage = () => {
        if (!streamMessageId) return;
        const id = streamMessageId;
        streamMessageId = null;
        setMessagesBySession((prev) => {
          const existing = prev[targetThreadId] ?? [];
          return { ...prev, [targetThreadId]: existing.filter((msg) => msg.id !== id) };
        });
      };

      try {
        const res = await fetch(`/api/threads/${targetThreadId}/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ content, model: activeModelId, useTavily: activeUseTavily }),
        });

        if (res.status === 401) {
          logout();
          removeThinkingMessage();
          removeStreamMessage();
          return;
        }
        if (!res.ok || !res.body) {
          const errorText = !res.ok ? await res.text() : "Resposta sem stream";
          removeThinkingMessage();
          throw new Error(errorText || "Falha ao iniciar streaming");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedText = "";
        let finalMessages: RawMessage[] | null = null;
        const traceEvents: TraceEvent[] = [];

        const ensureStreamMessage = () => {
          if (streamMessageId) return;
          streamMessageId = `stream-${Date.now()}`;
          const streamMessage: GenesisMessage = {
            id: streamMessageId,
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            modelId: activeModelId,
            usedTavily: activeUseTavily,
          };
          setMessagesBySession((prev) => {
            const existing = prev[targetThreadId] ?? [];
            const withoutThinking = existing.filter((msg) => msg.id !== thinkingMessageId);
            return { ...prev, [targetThreadId]: [...withoutThinking, streamMessage] };
          });
        };

        const updateStreamMessage = (text: string, trace?: TraceEvent[]) => {
          if (!streamMessageId) return;
          const id = streamMessageId;
          setMessagesBySession((prev) => {
            const existing = prev[targetThreadId] ?? [];
            return {
              ...prev,
              [targetThreadId]: existing.map((msg) =>
                msg.id === id ? { ...msg, content: text, ...(trace ? { trace } : {}) } : msg,
              ),
            };
          });
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() ?? "";

          for (const rawChunk of chunks) {
            const line = rawChunk.trim();
            if (!line.startsWith("data:")) continue;
            const jsonText = line.slice(line.indexOf("data:") + 5).trim();
            if (!jsonText) continue;
            let payload: StreamEventPayload;
            try {
              payload = JSON.parse(jsonText) as StreamEventPayload;
            } catch {
              continue;
            }
            if (payload.event === "trace") {
              traceEvents.push({
                type: payload.type as TraceEvent["type"],
                node: payload.node,
                tool: payload.tool,
                input: payload.input,
                output: payload.output,
                ts: payload.ts ?? Date.now(),
              });
            } else if (payload.event === "chunk") {
              const delta = payload.text ?? "";
              if (!delta) continue;
              ensureStreamMessage();
              accumulatedText += delta;
              updateStreamMessage(accumulatedText);
            } else if (payload.event === "final") {
              console.log("📦 Evento FINAL recebido:", payload);
              if (Array.isArray(payload.messages)) {
                finalMessages = payload.messages as RawMessage[];
                console.log("📝 Total de mensagens recebidas:", finalMessages.length);
              }
            } else if (payload.event === "error") {
              throw new Error(payload.detail || "Erro no streaming");
            }
          }
        }

        // Captura trace antes de qualquer limpeza
        const capturedTrace = traceEvents.length > 0 ? [...traceEvents] : null;

        removeThinkingMessage();
        removeStreamMessage();

        // Função auxiliar: aplica trace à última mensagem assistant da thread
        const applyTrace = (trace: TraceEvent[]) => {
          setMessagesBySession((prev) => {
            const msgs = prev[targetThreadId] ?? [];
            let lastIdx = -1;
            for (let i = msgs.length - 1; i >= 0; i--) {
              if (msgs[i]?.role === "assistant") { lastIdx = i; break; }
            }
            if (lastIdx === -1) return prev;
            const updated = [...msgs];
            updated[lastIdx] = { ...updated[lastIdx], trace };
            return { ...prev, [targetThreadId]: updated };
          });
        };

        if (finalMessages) {
          const normalized = finalMessages
            .map((m, index) => normalizeMessage(m, index))
            .filter((msg): msg is GenesisMessage => Boolean(msg))
            .filter((msg) => msg.role === "assistant");
          const annotatedFinal = annotateWithLastRun(targetThreadId, normalized);
          setMessagesBySession((prev) => {
            const existing = prev[targetThreadId] ?? [];
            const ids = new Set(existing.map((msg) => msg.id));
            const merged = [...existing];
            annotatedFinal.forEach((msg) => {
              if (!ids.has(msg.id)) {
                merged.push(msg);
              }
            });
            return { ...prev, [targetThreadId]: merged };
          });
          if (capturedTrace) applyTrace(capturedTrace);
          fetchThread(targetThreadId).catch(console.error);
        } else {
          await fetchThread(targetThreadId);
          if (capturedTrace) applyTrace(capturedTrace);
        }
      } catch (error) {
        console.error("Erro ao enviar mensagem:", error);
        removeThinkingMessage();
        removeStreamMessage();
        setMessagesBySession((prev) => {
          const existing = prev[targetThreadId] ?? [];
          const fallback: GenesisMessage = {
            id: `error-${Date.now()}`,
            role: "assistant",
            content: "Falha ao gerar resposta. Tente novamente.",
            timestamp: Date.now(),
          };
          return { ...prev, [targetThreadId]: [...existing, fallback] };
        });
      } finally {
        setIsSending(false);
      }
    },
    [currentSessionId, fetchThread, selectedModelId, token, logout, models],
  );

  const value = useMemo<GenesisUIState>(
    () => ({
      isLoading,
      isSending,
      models,
      selectedModelId,
      setSelectedModelId,
      useTavily,
      setUseTavily,
      sessions,
      currentSessionId,
      createSession,
      selectSession,
      renameSession,
      deleteSession,
      messagesBySession,
      sendMessage,
    }),
    [
      isLoading,
      isSending,
      models,
      selectedModelId,
      useTavily,
      sessions,
      currentSessionId,
      messagesBySession,
      createSession,
      selectSession,
      renameSession,
      deleteSession,
      sendMessage,
    ],
  );

  return <GenesisUIContext.Provider value={value}>{children}</GenesisUIContext.Provider>;
}

export function useGenesisUI() {
  const context = useContext(GenesisUIContext);
  if (!context) {
    throw new Error("useGenesisUI must be used within GenesisUIProvider");
  }
  return context;
}
