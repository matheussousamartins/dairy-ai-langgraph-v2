"use client";

import Link from "next/link";
import Image from "next/image";
import { FormEvent, useMemo, useState, useEffect, useRef, useCallback } from "react";
import clsx from "clsx";
import { useGenesisUI } from "@/state/useGenesisUI";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useAuth } from "@/state/useAuth";
import { Alert } from "@/components/ui/alert";
import { getAgentById } from "@/lib/agent-catalog";
import { type TraceEvent } from "@/state/useGenesisUI";


function TraceModal({ trace, onClose }: { trace: TraceEvent[]; onClose: () => void }) {
  const NODE_LABEL: Record<string, string> = {
    prepare: "Preparar contexto",
    agent: "LLM decidindo",
    tools: "Executando ferramenta",
    classify: "Classificar domínio",
    execute: "Executar sub-agentes",
    respond_direct: "Resposta direta",
    consolidate: "Consolidar respostas",
  };

  const TYPE_COLOR: Record<string, string> = {
    node_start: "text-[#05adca]",
    node_end:   "text-[#5c6383]",
    tool_call:  "text-amber-400",
    tool_result:"text-emerald-400",
  };

  const TYPE_PREFIX: Record<string, string> = {
    node_start:  "▶",
    node_end:    "◀",
    tool_call:   "⚙",
    tool_result: "✓",
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-end bg-black/40 backdrop-blur-sm p-4 sm:p-6"
      onClick={onClose}
    >
      <div
        className="relative flex h-[80vh] w-full max-w-2xl flex-col rounded-3xl border border-white/15 bg-[#05080f] shadow-[0_40px_100px_rgba(0,0,0,0.8)]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div>
            <p className="text-[10px] uppercase tracking-[0.4em] text-[#05adca]/70">Processo interno</p>
            <h3 className="text-base font-bold uppercase text-white" style={{ fontFamily: "var(--font-condensed)" }}>
              Log de Execução
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-white/15 px-3 py-1 text-[10px] uppercase tracking-[0.35em] text-[#9ba3c0] hover:border-white/30 hover:text-white"
          >
            Fechar
          </button>
        </div>

        {/* Eventos */}
        <div className="flex-1 overflow-y-auto px-5 py-4 font-mono text-xs">
          {trace.length === 0 ? (
            <p className="text-[#5c6383]">Nenhum evento registrado.</p>
          ) : (
            trace.map((ev, i) => (
              <div key={i} className="mb-3">
                <div className={clsx("flex items-center gap-2", TYPE_COLOR[ev.type] ?? "text-[#9ba3c0]")}>
                  <span>{TYPE_PREFIX[ev.type] ?? "•"}</span>
                  <span className="uppercase tracking-wider">
                    {ev.type === "node_start" || ev.type === "node_end"
                      ? `${ev.type === "node_start" ? "Início" : "Fim"} — ${NODE_LABEL[ev.node ?? ""] ?? ev.node}`
                      : ev.type === "tool_call"
                      ? `Busca RAG — ${ev.tool}`
                      : `Resultado — ${ev.tool}`}
                  </span>
                  <span className="ml-auto text-[#3a3f55]">
                    {new Date(ev.ts).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </span>
                </div>
                {ev.input && (
                  <div className="mt-1 ml-4 rounded-xl bg-[#0d1220] px-3 py-2 text-[11px] text-amber-300/80">
                    <span className="text-[#5c6383]">query › </span>{ev.input}
                  </div>
                )}
                {ev.output && (() => {
                  try {
                    const chunks = JSON.parse(ev.output) as Array<{ content: string; score?: number | null; source?: string }>;
                    if (Array.isArray(chunks)) {
                      return (
                        <div className="mt-1 ml-4 space-y-1.5">
                          {chunks.map((c, ci) => (
                            <div key={ci} className="rounded-xl bg-[#0d1220] px-3 py-2 text-[11px]">
                              <div className="mb-1 flex items-center gap-2 text-[10px] text-[#3a3f55]">
                                <span className="text-emerald-500/60">chunk {ci + 1}</span>
                                {c.score != null && <span>score: {c.score}</span>}
                                {c.source && <span className="truncate">{c.source}</span>}
                              </div>
                              <p className="text-emerald-300/70">{c.content}</p>
                            </div>
                          ))}
                        </div>
                      );
                    }
                  } catch { /* raw output */ }
                  return (
                    <pre className="mt-1 ml-4 max-h-36 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-[#0d1220] px-3 py-2 text-[11px] text-emerald-300/70">
                      {ev.output}
                    </pre>
                  );
                })()}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-white/10 px-5 py-3 text-[10px] uppercase tracking-[0.35em] text-[#3a3f55]">
          {trace.length} eventos · {trace.filter(e => e.type === "tool_call").length} buscas RAG
        </div>
      </div>
    </div>
  );
}

export function ChatPane() {
  const { isLoading, isSending, currentSessionId, sessions, messagesBySession, sendMessage, createSession } = useGenesisUI();
  const { token, isReady: authReady, isLoggingIn, loginError, login } = useAuth();
  const messages = useMemo(
    () => (currentSessionId ? messagesBySession[currentSessionId] ?? [] : []),
    [messagesBySession, currentSessionId],
  );
  const hasActiveSession = Boolean(currentSessionId);
  const currentSession = sessions.find((s) => s.id === currentSessionId);
  const sessionLabel = currentSession?.title ?? (currentSessionId ? currentSessionId.slice(0, 12) : "Selecione uma sessão");
  const [draft, setDraft] = useState("");
  const [passkeyInput, setPasskeyInput] = useState("");
  const [localLoginError, setLocalLoginError] = useState<string | null>(null);
  const [activeTrace, setActiveTrace] = useState<TraceEvent[] | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const [userHasScrolled, setUserHasScrolled] = useState(false);
  const lastMessageCountRef = useRef(0);

  const handleSubmit = useCallback(async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || isLoading || isSending || !token || !hasActiveSession) return;

    // Limpa o draft imediatamente para feedback visual
    setDraft("");

    // Reseta o controle de scroll ao enviar mensagem
    setUserHasScrolled(false);

    try {
      await sendMessage(trimmed);
    } catch (error) {
      // Se houver erro, restaura o draft
      setDraft(trimmed);
      console.error("Erro ao enviar mensagem:", error);
    }
  }, [draft, hasActiveSession, isLoading, isSending, token, sendMessage]);

  const handleCreateSession = useCallback(() => {
    if (isLoading || isSending) return;
    createSession().catch(console.error);
  }, [createSession, isLoading, isSending]);

  async function handleLoginSubmit(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const trimmed = passkeyInput.trim();
    if (!trimmed) {
      setLocalLoginError("Informe a passkey para continuar.");
      return;
    }
    setLocalLoginError(null);
    try {
      await login(trimmed);
      setPasskeyInput("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Falha ao autenticar";
      setLocalLoginError(message);
    }
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        handleSubmit();
      }
    }

    const textarea = textareaRef.current;
    if (textarea) {
      textarea.addEventListener("keydown", handleKeyDown);
      return () => textarea.removeEventListener("keydown", handleKeyDown);
    }
  }, [handleSubmit]);

  // Detecta qualquer interação de scroll do usuário
  useEffect(() => {
    const mainElement = mainRef.current;
    if (!mainElement) return;

    function handleUserScroll(event: Event) {
      const target = event.target as HTMLElement;
      if (!target) return;
      
      // Verifica se o usuário está scrollando para cima
      const { scrollTop, scrollHeight, clientHeight } = target;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      
      // Se não está perto do fim, marca que o usuário scrollou
      if (!isNearBottom) {
        setUserHasScrolled(true);
      }
    }

    // Detecta scroll via roda do mouse, touch, ou barra de rolagem
    mainElement.addEventListener('scroll', handleUserScroll, { passive: true });
    mainElement.addEventListener('wheel', handleUserScroll, { passive: true });
    mainElement.addEventListener('touchmove', handleUserScroll, { passive: true });
    
    return () => {
      mainElement.removeEventListener('scroll', handleUserScroll);
      mainElement.removeEventListener('wheel', handleUserScroll);
      mainElement.removeEventListener('touchmove', handleUserScroll);
    };
  }, []);

  // Auto-scroll quando novas mensagens chegam
  useEffect(() => {
    const hasNewMessages = messages.length > lastMessageCountRef.current;
    lastMessageCountRef.current = messages.length;
    
    // Faz scroll se: não scrollou manualmente OU se há novas mensagens
    if (!userHasScrolled || hasNewMessages) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, userHasScrolled]);

  // Reseta o controle quando troca de sessão
  useEffect(() => {
    setUserHasScrolled(false);
    lastMessageCountRef.current = 0;
  }, [currentSessionId]);

  if (!authReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-[#080d18] via-[#0c1324] to-[#05080f] px-4">
        <p className="text-xs uppercase tracking-[0.45em] text-[#7f8baf]">Sincronizando credenciais…</p>
      </div>
    );
  }

  if (!token) {
    const authMessage = localLoginError || loginError;
    return (
      <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-[#05080f] via-[#0b1324] to-[#111a2d] px-4 py-10 sm:px-6 lg:px-10">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_15%_10%,rgba(32,51,109,0.22),transparent_55%)]" />
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_80%_8%,rgba(16,134,173,0.2),transparent_50%)]" />
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,0.05),transparent)]" />

        <div className="relative mx-auto flex max-w-4xl flex-col items-center gap-8 text-center">
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center justify-center rounded-3xl bg-black/40 p-2 shadow-[0_20px_60px_rgba(16,134,173,0.35)] ring-1 ring-white/10">
              <Image
                src="/commandix-logo.png"
                alt="Hawk"
                width={140}
                height={140}
                priority
                className="h-28 w-28 rounded-2xl drop-shadow-[0_20px_60px_rgba(16,134,173,0.45)]"
                style={{ objectFit: "contain" }}
              />
            </div>
            <div className="space-y-2">
              <h1
                className="text-4xl font-black uppercase leading-tight text-white sm:text-5xl"
                style={{ fontFamily: "var(--font-condensed)" }}
              >
                Commandix Tech
              </h1>
            </div>
          </div>

          <section className="w-full max-w-xl text-left">
            <div className="rounded-[32px] border border-white/15 bg-gradient-to-br from-[#090f1c]/95 via-[#101a32]/95 to-[#05080f]/95 p-6 text-[#dfdecf] shadow-[0_40px_100px_rgba(0,0,0,0.75)] ring-1 ring-white/10 sm:p-8">
              <div className="flex items-center justify-between">
                <p className="text-[11px] uppercase tracking-[0.5em] text-[#05adca]/70">Acesso restrito</p>
                <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-3 py-1 text-[11px] uppercase tracking-[0.35em] text-[#9ba3c0]">
                  <span className="h-2 w-2 rounded-full bg-[#58d38f]" aria-hidden />
                  <Link
                    href="https://www.rhawk.pro/comunidade"
                    target="_blank"
                    rel="noreferrer"
                    className="transition hover:text-white"
                  >
                    Commandix
                  </Link>
                </span>
              </div>
              <h2
                className="mt-3 text-3xl font-bold uppercase text-white sm:text-4xl"
                style={{ fontFamily: "var(--font-condensed)" }}
              >
                DairyApp
              </h2>
              <p className="mt-3 text-sm text-[#9ba3c0]">
                Insira o passkey gerada. Ao liberar, você cai direto no console em tempo real.
              </p>
              <form onSubmit={handleLoginSubmit} className="mt-6 space-y-4 sm:mt-8">
                <label className="block text-[11px] uppercase tracking-[0.3em] text-[#7f8baf]">
                  Senha de acesso
                </label>
                <input
                  type="password"
                  value={passkeyInput}
                  onChange={(event) => setPasskeyInput(event.target.value)}
                  placeholder="********"
                  className="w-full rounded-2xl border border-white/15 bg-[#050b16] px-4 py-3 text-sm text-white placeholder:text-[#5f6785] focus:border-[#1086ad] focus:outline-none focus:ring-1 focus:ring-[#1086ad]"
                />
                {authMessage ? (
                  <Alert variant="error" className="mt-3">
                    <span className="text-sm">{authMessage}</span>
                  </Alert>
                ) : null}
                <Button
                  type="submit"
                  disabled={isLoggingIn || !passkeyInput.trim()}
                  className="w-full justify-center py-4 text-sm"
                >
                  {isLoggingIn ? "Validando..." : "Liberar console"}
                </Button>
              </form>
            </div>
          </section>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-[#090e1a]">
      {/* Overlays decorativos fixos — fora do scroll container para evitar cortes visuais */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_15%,rgba(32,51,109,0.22),transparent_60%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_80%_0%,rgba(242,101,49,0.12),transparent_55%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,0.04),transparent)]" />
      <main
        ref={mainRef}
        className="relative flex flex-1 flex-col gap-5 overflow-y-auto px-4 py-6 text-[#dfdecf] sm:px-6 lg:px-10"
      >
        <div className="relative flex-1 space-y-4">
          {!hasActiveSession ? (
            <div className="flex h-full items-center justify-center">
              <Card className="w-full max-w-3xl border border-white/15 bg-[rgba(9,14,26,0.92)] text-white shadow-[0_45px_120px_rgba(0,0,0,0.65)]">
                <CardHeader className="text-center">
                  <p className="text-[11px] uppercase tracking-[0.4em] text-[#05adca]/70">Seleção de Sessão</p>
                  <h2 className="text-3xl font-bold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                    Iniciar Consulta
                  </h2>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <button
                      type="button"
                      onClick={handleCreateSession}
                      disabled={isLoading}
                      className={clsx(
                        "flex h-44 flex-col justify-between rounded-3xl border px-5 py-4 text-left transition-all",
                        "border-[#1086ad]/70 bg-[#1086ad]/12 text-white shadow-[0_30px_70px_rgba(6,12,24,0.65)] disabled:opacity-50",
                      )}
                    >
                      <span className="text-xl font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                        Nova Sessão
                      </span>
                      <span className="text-xs text-[#f5c7b4]">Inicie uma nova conversa com um agente especializado em laticínios.</span>
                    </button>
                    <Link
                      href="/history"
                      className="flex h-44 flex-col justify-between rounded-3xl border border-white/15 px-5 py-4 text-left text-[#dfdecf] transition hover:border-white/40 hover:bg-white/5"
                    >
                      <span className="text-xl font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                        Ver Histórico
                      </span>
                      <span className="text-xs text-[#7f8baf]">Retome ou revise consultas realizadas anteriormente.</span>
                    </Link>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : isLoading ? (
            <Card className="border border-white/12 bg-white/5 text-center text-[#9ba3c0]">
              <CardHeader>Carregando sessões</CardHeader>
              <CardContent>Aguarde enquanto conectamos ao LangGraph.</CardContent>
            </Card>
          ) : messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <p
                className="select-none text-4xl font-black uppercase tracking-[0.18em] text-white/5"
                style={{ fontFamily: "var(--font-condensed)" }}
              >
                Dairy AI
              </p>
            </div>
          ) : (
            messages.map((message) => {
              const isAssistant = message.role === "assistant";
              const isThinking = message.content === "Pensando...";
              return (
                <article
                  key={message.id}
                  className={clsx(
                    "w-full max-w-full rounded-[28px] border px-5 py-5 text-sm leading-relaxed shadow-[0_25px_70px_rgba(0,0,0,0.55)] transition-all sm:px-6 md:max-w-3xl",
                    isAssistant
                      ? "border-white/10 bg-gradient-to-br from-[#111a2d]/90 to-[#0a0f1d]/90 text-[#f6f7fb]"
                      : "border-[#1086ad]/60 bg-[#1086ad]/12 text-[#ecf6ff] ring-1 ring-[#1086ad]/30 sm:ml-auto",
                  )}
                >
                  <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.35em] text-[#7f8baf]">
                    <span>{message.role === "assistant" ? "DairyApp" : "Operador"}</span>
                    <span className="text-[#5c6383]">{new Date(message.timestamp).toLocaleTimeString()}</span>
                  </div>
                  {isAssistant ? (
                    isThinking ? (
                      <div className="flex items-center gap-2 italic text-[#9ba3c0]">
                        <span className="animate-pulse">Pensando...</span>
                      </div>
                    ) : (
                      <div className="markdown-body">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ ...props }) => (
                              <a
                                {...props}
                                className="font-semibold text-[#05adca] underline decoration-[#05adca]/60 underline-offset-4 hover:text-white"
                                target="_blank"
                                rel="noreferrer"
                              />
                            ),
                            code: ({ className, children, ...props }) => {
                              const isInline = !className?.includes('language-');
                              return isInline ? (
                                <code
                                  className={clsx(
                                    "rounded bg-[#1a2236] px-1.5 py-0.5 text-[13px] text-[#f6f7fb]",
                                    className,
                                  )}
                                  {...props}
                                >
                                  {children}
                                </code>
                              ) : (
                                <pre
                                  className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050912] p-4 text-[13px] text-[#dfdecf]"
                                >
                                  <code className={className} {...props}>{children}</code>
                                </pre>
                              );
                            },
                            li: ({ ...props }) => <li className="pl-1" {...props} />,
                          }}
                        >
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    )
                  ) : (
                    <p
                      className="whitespace-pre-wrap break-words text-[15px] text-[#e6f4ff]"
                      style={{ fontFamily: "var(--font-sans)" }}
                    >
                      {message.content}
                    </p>
                  )}
                <div className="mt-3 flex items-center gap-3 text-[10px] uppercase tracking-[0.35em] text-[#7f8baf]">
                  {message.modelId ? (
                    <span>Agente: {getAgentById(message.modelId)?.label ?? message.modelId}</span>
                  ) : null}
                  {isAssistant && message.trace && message.trace.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setActiveTrace(message.trace!)}
                      className="ml-auto rounded-full border border-white/15 bg-white/5 px-2.5 py-1 font-mono text-[10px] text-[#5c6383] transition hover:border-[#05adca]/40 hover:text-[#05adca]"
                      title="Ver log de execução"
                    >
                      {"</>"}
                    </button>
                  )}
                </div>
              </article>
              );
            })
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {activeTrace && (
        <TraceModal trace={activeTrace} onClose={() => setActiveTrace(null)} />
      )}

      {hasActiveSession ? (
        <footer className="border-t border-white/5 bg-[#090f1c]/80 px-4 py-5 backdrop-blur-2xl sm:px-6 lg:px-10">
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-3 sm:flex-row sm:items-end sm:gap-4">
            <div className="flex-1">
              <label className="mb-2 block text-[11px] uppercase tracking-[0.35em] text-[#7f8baf]">
                Digite sua pergunta
              </label>
              <textarea
                ref={textareaRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder={isLoading ? "Carregando..." : "Ex: Qual o limite de coliformes para queijo minas frescal?"}
                className="h-28 w-full resize-none rounded-2xl border border-white/15 bg-gradient-to-br from-[#070b14] to-[#111b33] px-5 py-4 text-sm text-white placeholder:text-[#5c6383] shadow-[0_35px_80px_rgba(0,0,0,0.55)] focus:border-[#1086ad] focus:outline-none focus:ring-2 focus:ring-[#1086ad]/70 sm:h-32 lg:h-[110px]"
                disabled={isLoading}
              />
            </div>
            <div className="flex w-full items-end sm:w-auto sm:pt-[30px]">
              <Button
                type="submit"
                disabled={isLoading || isSending || !draft.trim()}
                className="h-12 w-full px-6 text-[11px] sm:h-[110px] sm:w-auto sm:px-10"
              >
                Enviar
              </Button>
            </div>
          </form>
          <div className="mt-3 text-[11px] uppercase tracking-[0.3em] text-[#7f8baf]">
            Consulta ativa: <span className="text-white">{sessionLabel}</span>
          </div>
        </footer>
      ) : null}
    </div>
  );
}
