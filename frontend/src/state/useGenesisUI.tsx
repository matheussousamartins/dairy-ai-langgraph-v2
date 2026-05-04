"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "./useAuth";
import {
  isWeakThreadPrompt,
  shouldReplaceSessionTitle,
  summarizeThreadTitleFromMessages,
  summarizeThreadTitleFromText,
} from "@/lib/thread-title";

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
  turnId?: string;
  modelId?: string;
  agentId?: string;
  usedTavily?: boolean;
  trace?: TraceEvent[];
}

interface RawMessage {
  id?: string;
  type?: string;
  content?: unknown;
  turn_id?: string;
  trace?: TraceEvent[];
  response_metadata?: {
    model_name?: string;
    agent_id?: string;
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
  summary?: {
    title?: string;
    preview?: string;
    message_count?: number;
    question?: string;
    last_agent_id?: string;
  };
  values?: {
    messages?: RawMessage[];
  };
}

interface ModelsResponse {
  default_model?: string;
  models?: Array<{
    id: string;
    label: string;
    provider?: string;
    description?: string;
    family?: string;
    family_subtitle?: string;
    compatibility_status?: "ready" | "requires_adapter";
    compatibility_message?: string;
    setup_hint?: string;
    selectable?: boolean;
    input_cost?: number;
    output_cost?: number;
  }>;
}

interface AgentsResponse {
  agents?: Array<{
    id: string;
    label: string;
    endpoint: string;
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
  const agentId = message.response_metadata?.agent_id;
  
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
    turnId: message.turn_id,
    modelId,
    agentId,
    usedTavily: Boolean(message.tool_calls && (message.tool_calls as unknown[]).length > 0),
    trace: Array.isArray(message.trace) ? message.trace : undefined,
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
  const summarizedTitle = summarizeThreadTitleFromMessages(
    normalized.map((message) => ({ role: message.role, content: message.content })),
    thread.thread_id,
  );
  const title =
    thread.summary?.title && !isWeakThreadPrompt(thread.summary.title)
      ? thread.summary.title
      : summarizedTitle;
  const session: GenesisSession = {
    id: thread.thread_id,
    title,
    createdAt: threadCreatedAt,
    preview: thread.summary?.preview ?? messages.at(-1)?.content?.toString() ?? "",
    messageCount: thread.summary?.message_count ?? normalized.length,
    question: thread.summary?.question ?? firstUser?.content ?? "",
    lastAgentId: thread.summary?.last_agent_id,
  };
  return { session, messages: normalized };
}

export interface GenesisSession {
  id: string;
  title: string;
  createdAt: number;
  preview?: string;
  messageCount?: number;
  question?: string;
  lastAgentId?: string;
}

export interface ModelOption {
  id: string;
  label: string;
  provider?: string;
  description?: string;
  family?: string;
  familySubtitle?: string;
  compatibilityStatus?: "ready" | "requires_adapter";
  compatibilityMessage?: string;
  setupHint?: string;
  selectable?: boolean;
  inputCost: number;
  outputCost: number;
}

export interface AgentOption {
  id: string;
  label: string;
  endpoint: string;
  inputCost: number;
  outputCost: number;
}

interface GenesisCatalogState {
  isLoading: boolean;
  isSending: boolean;
  agents: AgentOption[];
  selectedAgentId: string;
  setSelectedAgentId: (id: string) => void;
  models: ModelOption[];
  defaultModelId: string;
  selectedModelId: string;
  setSelectedModelId: (id: string) => void;
  useTavily: boolean;
  setUseTavily: (value: boolean) => void;
}

interface GenesisConversationState {
  isLoading: boolean;
  isSending: boolean;
  sessions: GenesisSession[];
  currentSessionId: string;
  createSession: () => Promise<string | undefined>;
  selectSession: (id: string) => Promise<void>;
  renameSession: (id: string, title: string) => void;
  deleteSession: (id: string) => void;
  messagesBySession: Record<string, GenesisMessage[]>;
  sendMessage: (content: string) => Promise<void>;
}

type GenesisUIState = GenesisCatalogState & GenesisConversationState;

const GenesisCatalogContext = createContext<GenesisCatalogState | null>(null);
const GenesisConversationContext = createContext<GenesisConversationState | null>(null);

export function GenesisUIProvider({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSending, setIsSending] = useState<boolean>(false);
  const [agents, setAgents] = useState<AgentOption[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string>("");
  const [models, setModels] = useState<ModelOption[]>([]);
  const [defaultModelId, setDefaultModelId] = useState<string>("");
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [useTavily, setUseTavily] = useState<boolean>(true);
  const [sessions, setSessions] = useState<GenesisSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [messagesBySession, setMessagesBySession] = useState<Record<string, GenesisMessage[]>>({});
  const { token, isReady: authReady, logout } = useAuth();
  const selectedAgentRef = useRef<string>("");
  const selectedModelRef = useRef<string>("");
  const useTavilyRef = useRef<boolean>(useTavily);
  const lastRunSettingsRef = useRef<Record<string, { agentId: string; modelId: string; useTavily: boolean }>>({});

  useEffect(() => {
    selectedAgentRef.current = selectedAgentId;
  }, [selectedAgentId]);

  useEffect(() => {
    selectedModelRef.current = selectedModelId;
  }, [selectedModelId]);

  useEffect(() => {
    useTavilyRef.current = useTavily;
  }, [useTavily]);

  const loadAgents = useCallback(async () => {
    if (!token) return;
    const res = await fetch("/api/agents", {
      cache: "no-store",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (res.status === 401) {
      logout();
      return;
    }
    if (!res.ok) return;
    const data = (await res.json()) as AgentsResponse;
    const mapped: AgentOption[] = (data.agents ?? []).map((agent) => ({
      id: agent.id,
      label: agent.label,
      endpoint: agent.endpoint,
      inputCost: agent.input_cost ?? 0,
      outputCost: agent.output_cost ?? 0,
    }));
    setAgents(mapped);
    setSelectedAgentId((prev) => (prev ? prev : mapped[0]?.id ?? ""));
  }, [token, logout]);

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
      provider: model.provider,
      description: model.description,
      family: model.family,
      familySubtitle: model.family_subtitle,
      compatibilityStatus: model.compatibility_status,
      compatibilityMessage: model.compatibility_message,
      setupHint: model.setup_hint,
      selectable: model.selectable,
      inputCost: model.input_cost ?? 0,
      outputCost: model.output_cost ?? 0,
    }));
    const firstSelectableId = mapped.find((model) => model.selectable !== false)?.id ?? mapped[0]?.id ?? "";
    setModels(mapped);
    setDefaultModelId(data.default_model ?? firstSelectableId);
    setSelectedModelId((prev) => {
      if (prev && mapped.some((model) => model.id === prev && model.selectable !== false)) {
        return prev;
      }
      return firstSelectableId;
    });
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
      agentId: last.agentId,
      usedTavily: last.useTavily,
    };
    return cloned;
  };

  const purgeMissingThread = useCallback((threadId: string) => {
    delete lastRunSettingsRef.current[threadId];
    setSessions((prev) => prev.filter((session) => session.id !== threadId));
    setCurrentSessionId((prevId) => (prevId === threadId ? "" : prevId));
    setMessagesBySession((prev) => {
      const next = { ...prev };
      delete next[threadId];
      return next;
    });
  }, []);

  const fetchThread = useCallback(
    async (threadId: string) => {
      if (!token) return false;
      const res = await fetch(`/api/threads/${threadId}`, {
        headers: { Authorization: `Bearer ${token}` },
        cache: "no-store",
      });
      if (res.status === 401) {
        logout();
        return false;
      }
      if (res.status === 404) {
        purgeMissingThread(threadId);
        return false;
      }
      if (!res.ok) return false;
      const data = await res.json();
      const thread = data.thread;
      if (!thread) return false;
      const { messages, session } = normalizeThread(thread);
      setMessagesBySession((prev) => {
        const existingMsgs = prev[threadId] ?? [];
        const traceById = new Map(existingMsgs.filter((m) => m.trace).map((m) => [m.id, m.trace!]));
        const annotated = annotateWithLastRun(threadId, messages);
        const withTraces = annotated.map((m) => {
          const t = traceById.get(m.id);
          return t ? { ...m, trace: t } : m;
        });
        return { ...prev, [threadId]: withTraces };
      });
      setSessions((prev) => {
        const exists = prev.find((s) => s.id === threadId);
        if (exists) {
          return prev.map((s) => (s.id === threadId ? { ...s, ...session } : s));
        }
        return [session, ...prev];
      });
      return true;
    },
    [token, logout, purgeMissingThread],
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
    let sessionToHydrate: string | null = null;
    setCurrentSessionId((prev) => {
      if (prev && nextSessions.some((session) => session.id === prev)) {
        sessionToHydrate = prev;
        return prev;
      }
      return "";
    });

    if (sessionToHydrate) {
      await fetchThread(sessionToHydrate).catch((error) => {
        console.error("Falha ao hidratar thread ativa", sessionToHydrate, error);
      });
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
        await Promise.all([loadAgents(), loadModels(), loadThreads()]);
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
  }, [authReady, token, loadAgents, loadModels, loadThreads]);

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

      const activeAgentId = selectedAgentRef.current || selectedAgentId || agents[0]?.id || "";
      const activeModelId = selectedModelRef.current || selectedModelId || models[0]?.id || "";
      if (!activeAgentId || !activeModelId) {
        console.error("Agente ou modelo não selecionado para envio da mensagem.");
        return;
      }
      const activeUseTavily = useTavilyRef.current;
      lastRunSettingsRef.current[threadId] = {
        agentId: activeAgentId,
        modelId: activeModelId,
        useTavily: activeUseTavily,
      };

      const optimistic: GenesisMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content,
        timestamp: Date.now(),
        modelId: activeModelId,
        agentId: activeAgentId,
        usedTavily: activeUseTavily,
      };

      setMessagesBySession((prev) => {
        const existing = prev[threadId] ?? [];
        return { ...prev, [threadId]: [...existing, optimistic] };
      });
      const nextTitle = summarizeThreadTitleFromText(content);
      if (nextTitle) {
        setSessions((prev) =>
          prev.map((session) =>
            session.id === threadId && shouldReplaceSessionTitle(session.title)
              ? { ...session, title: nextTitle, question: content }
              : session,
          ),
        );
      }
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

      const materializeAssistantMessage = (text: string, trace?: TraceEvent[] | null) => {
        const trimmed = text.trim();
        if (!trimmed) return;

        setMessagesBySession((prev) => {
          const existing = prev[targetThreadId] ?? [];
          const withoutThinking = existing.filter((msg) => msg.id !== thinkingMessageId);

          if (streamMessageId) {
            const id = streamMessageId;
            return {
              ...prev,
              [targetThreadId]: withoutThinking.map((msg) =>
                msg.id === id
                  ? {
                      ...msg,
                      content: trimmed,
                      modelId: activeModelId,
                      agentId: activeAgentId,
                      usedTavily: activeUseTavily,
                      ...(trace ? { trace } : {}),
                    }
                  : msg,
              ),
            };
          }

          const fallbackAssistant: GenesisMessage = {
            id: `assistant-local-${Date.now()}`,
            role: "assistant",
            content: trimmed,
            timestamp: Date.now(),
            modelId: activeModelId,
            agentId: activeAgentId,
            usedTavily: activeUseTavily,
            ...(trace ? { trace } : {}),
          };

          return { ...prev, [targetThreadId]: [...withoutThinking, fallbackAssistant] };
        });
      };

      try {
        const res = await fetch(`/api/threads/${targetThreadId}/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            content,
            model: activeModelId,
            agentId: activeAgentId,
            useTavily: activeUseTavily,
          }),
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
          if (res.status === 404 && errorText.includes("Thread not found")) {
            purgeMissingThread(targetThreadId);
          }
          throw new Error(errorText || "Falha ao iniciar streaming");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let accumulatedText = "";
        let finalMessages: RawMessage[] | null = null;
        const traceEvents: TraceEvent[] = [];
        const updateThinkingMessage = (trace: TraceEvent[]) => {
          setMessagesBySession((prev) => {
            const existing = prev[targetThreadId] ?? [];
            return {
              ...prev,
              [targetThreadId]: existing.map((msg) =>
                msg.id === thinkingMessageId ? { ...msg, trace: [...trace] } : msg,
              ),
            };
          });
        };
        const processStreamRecord = (line: string) => {
          if (!line.startsWith("data:")) return;
          const jsonText = line.slice(line.indexOf("data:") + 5).trim();
          if (!jsonText) return;
          let payload: StreamEventPayload;
          try {
            payload = JSON.parse(jsonText) as StreamEventPayload;
          } catch {
            return;
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
            if (streamMessageId) {
              updateStreamMessage(accumulatedText, [...traceEvents]);
            } else {
              updateThinkingMessage(traceEvents);
            }
          } else if (payload.event === "chunk") {
            const delta = payload.text ?? "";
            if (!delta) return;
            ensureStreamMessage();
            accumulatedText += delta;
            updateStreamMessage(accumulatedText, traceEvents.length > 0 ? [...traceEvents] : undefined);
          } else if (payload.event === "final") {
            if (Array.isArray(payload.messages)) {
              finalMessages = payload.messages as RawMessage[];
            }
          } else if (payload.event === "error") {
            throw new Error(payload.detail || "Erro no streaming");
          }
        };

        const ensureStreamMessage = () => {
          if (streamMessageId) return;
          streamMessageId = `stream-${Date.now()}`;
          const streamMessage: GenesisMessage = {
            id: streamMessageId,
            role: "assistant",
            content: "",
            timestamp: Date.now(),
            modelId: activeModelId,
            agentId: activeAgentId,
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
            if (!line) continue;
            processStreamRecord(line);
          }
        }

        const trailing = buffer.trim();
        if (trailing) {
          processStreamRecord(trailing);
        }

        // Captura trace antes de qualquer limpeza
        const capturedTrace = traceEvents.length > 0 ? [...traceEvents] : null;

        removeThinkingMessage();

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

        const finalMessagesList: RawMessage[] = finalMessages ?? [];

        if (finalMessagesList.length > 0) {
          const normalized = finalMessagesList
            .map((m, index) => normalizeMessage(m, index))
            .filter((msg): msg is GenesisMessage => Boolean(msg))
            .filter((msg) => msg.role === "assistant");
          const annotatedFinal = annotateWithLastRun(targetThreadId, normalized);
          if (annotatedFinal.length > 0) {
            removeStreamMessage();
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
          } else {
            materializeAssistantMessage(accumulatedText, capturedTrace);
          }
          if (capturedTrace) applyTrace(capturedTrace);
          fetchThread(targetThreadId).catch(console.error);
        } else {
          materializeAssistantMessage(accumulatedText, capturedTrace);
          try {
            await fetchThread(targetThreadId);
            if (capturedTrace) applyTrace(capturedTrace);
          } catch (error) {
            console.error("Falha ao recarregar thread apos stream:", error);
            if (streamMessageId) {
              updateStreamMessage(accumulatedText, capturedTrace ?? undefined);
            }
          } finally {
            removeStreamMessage();
          }
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
    [agents, currentSessionId, fetchThread, logout, models, purgeMissingThread, selectedAgentId, selectedModelId, token],
  );

  const catalogValue = useMemo<GenesisCatalogState>(
    () => ({
      isLoading,
      isSending,
      agents,
      selectedAgentId,
      setSelectedAgentId,
      models,
      defaultModelId,
      selectedModelId,
      setSelectedModelId,
      useTavily,
      setUseTavily,
    }),
    [
      isLoading,
      isSending,
      agents,
      selectedAgentId,
      models,
      defaultModelId,
      selectedModelId,
      useTavily,
    ],
  );

  const conversationValue = useMemo<GenesisConversationState>(
    () => ({
      isLoading,
      isSending,
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
      sessions,
      currentSessionId,
      createSession,
      selectSession,
      renameSession,
      deleteSession,
      messagesBySession,
      sendMessage,
    ],
  );

  return (
    <GenesisCatalogContext.Provider value={catalogValue}>
      <GenesisConversationContext.Provider value={conversationValue}>
        {children}
      </GenesisConversationContext.Provider>
    </GenesisCatalogContext.Provider>
  );
}

export function useGenesisCatalog() {
  const context = useContext(GenesisCatalogContext);
  if (!context) {
    throw new Error("useGenesisCatalog must be used within GenesisUIProvider");
  }
  return context;
}

export function useGenesisConversation() {
  const context = useContext(GenesisConversationContext);
  if (!context) {
    throw new Error("useGenesisConversation must be used within GenesisUIProvider");
  }
  return context;
}

export function useGenesisUI(): GenesisUIState {
  const catalog = useGenesisCatalog();
  const conversation = useGenesisConversation();
  return { ...catalog, ...conversation };
}


