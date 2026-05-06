"use client";

import Link from "next/link";
import Image from "next/image";
import dynamic from "next/dynamic";
import {
  FormEvent,
  type ReactNode,
  type RefObject,
  memo,
  useMemo,
  useState,
  useEffect,
  useRef,
  useCallback,
  type ComponentProps,
} from "react";
import clsx from "clsx";
import { useGenesisCatalog, useGenesisConversation } from "@/state/useGenesisUI";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Activity, Check, ClipboardCheck, Minus, ThumbsDown, ThumbsUp } from "lucide-react";
import { useAuth } from "@/state/useAuth";
import { Alert } from "@/components/ui/alert";
import { getAgentById } from "@/lib/agent-catalog";
import { type GenesisMessage, type ModelOption, type TraceEvent } from "@/state/useGenesisUI";
import { useThreadTesting, type TestErrorCategory, type TestVerdict } from "@/state/useThreadTesting";
const LazyEvaluationModal = dynamic(
  () => import("@/components/app/EvaluationModal").then((module) => module.EvaluationModal),
  { ssr: false },
);

const LazyTraceModal = dynamic(
  () => import("@/components/app/TraceModal").then((module) => module.TraceModal),
  { ssr: false },
);

const MdLink = ({ ...props }: ComponentProps<"a">) => (
  <a
    {...props}
    className="font-semibold text-[#b06fff] underline decoration-[#b06fff]/60 underline-offset-4 hover:text-white"
    target="_blank"
    rel="noreferrer"
  />
);
const MdCode = ({ className, children, ...props }: { className?: string; children?: ReactNode }) => {
  const isInline = !className?.includes("language-");
  return isInline ? (
    <code
      className={clsx("rounded bg-[#1a2236] px-1.5 py-0.5 text-[13px] text-[#f6f7fb]", className)}
      {...props}
    >
      {children}
    </code>
  ) : (
    <pre className="overflow-x-auto rounded-2xl border border-white/10 bg-[#050912] p-4 text-[13px] text-[#dfdecf]">
      <code className={className} {...props}>
        {children}
      </code>
    </pre>
  );
};
const MdLi = ({ ...props }: ComponentProps<"li">) => <li className="pl-1" {...props} />;

const markdownComponents = { a: MdLink, code: MdCode, li: MdLi };

function deriveThinkingCopy(trace?: TraceEvent[]) {
  const lastEvent = trace?.slice(-1)[0];

  if (!lastEvent) {
    return {
      title: "Pensando",
      detail: "Entendendo a pergunta e preparando a melhor resposta.",
      accent: "Análise inicial",
    };
  }

  if (lastEvent.type === "tool_call") {
    return {
      title: "Consultando contexto",
      detail: lastEvent.tool
        ? `Usando ${lastEvent.tool} para enriquecer a resposta.`
        : "Buscando informações complementares para responder com mais precisão.",
      accent: "Ferramenta em uso",
    };
  }

  if (lastEvent.type === "tool_result") {
    return {
      title: "Organizando informações",
      detail: "Consolidando os dados encontrados antes de responder.",
      accent: "Contexto recebido",
    };
  }

  if (lastEvent.type === "node_start") {
    const nodeName = (lastEvent.node ?? "").toLowerCase();
    if (nodeName.includes("orques")) {
      return {
        title: "Selecionando agente",
        detail: "Escolhendo o especialista mais adequado para esta pergunta.",
        accent: "Orquestração",
      };
    }
    return {
      title: "Processando resposta",
      detail: lastEvent.node
        ? `Executando a etapa ${lastEvent.node}.`
        : "Processando a melhor forma de responder.",
      accent: "Execução em andamento",
    };
  }

  if (lastEvent.type === "node_end") {
    return {
      title: "Finalizando resposta",
      detail: "Ajustando os últimos detalhes antes de mostrar a resposta.",
      accent: "Resposta chegando",
    };
  }

  return {
    title: "Pensando",
    detail: "Preparando a resposta com base no contexto disponível.",
    accent: "Processando",
  };
}

// Helper: Is the user message that preceded an assistant message a real question (not a greeting)?
function isPrevUserSubstantial(messages: GenesisMessage[], assistantMessageId: string): boolean {
  const idx = messages.findIndex((m) => m.id === assistantMessageId);
  if (idx <= 0) return false;
  for (let i = idx - 1; i >= 0; i--) {
    const m = messages[i];
    if (m?.role === "user") return m.content.trim().length >= 15;
  }
  return false;
}

const AssistantMessageBody = memo(function AssistantMessageBody({
  content,
  isThinking,
  trace,
}: {
  content: string;
  isThinking: boolean;
  trace?: TraceEvent[];
}) {
  if (isThinking) {
    const copy = deriveThinkingCopy(trace);
    return (
      <div className="flex items-start gap-3 text-[#aeb4c6]">
        <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[#8b3dff]/20 bg-[#8b3dff]/10 text-[#c7a6ff] shadow-[0_0_24px_rgba(139,61,255,0.14)]">
          <Activity className="h-4 w-4 animate-pulse" />
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[15px] italic text-[#cfd4e3]">{copy.title}</span>
            <span className="inline-flex items-center gap-1 text-[#8f96ab]">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b06fff]" />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b06fff]" style={{ animationDelay: "120ms" }} />
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b06fff]" style={{ animationDelay: "240ms" }} />
            </span>
          </div>
          <p className="text-[12px] leading-5 text-[#8f96ab]">{copy.detail}</p>
          <span className="inline-flex rounded-full border border-white/10 bg-white/[0.03] px-2.5 py-1 text-[10px] uppercase tracking-[0.28em] text-[#8f96ab]">
            {copy.accent}
          </span>
        </div>
      </div>
    );
  }
  return (
    <div className="markdown-body text-[15px] leading-[1.75] text-[#edf0f8] [&_li]:text-[15px] [&_p]:text-[15px]">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {content}
      </ReactMarkdown>
    </div>
  );
});

const MessageCard = memo(function MessageCard({
  message,
  models,
  onOpenTrace,
  evaluationVerdict,
  onEvaluate,
  isEvaluating,
  isSubstantialQuery,
}: {
  message: GenesisMessage;
  models: ModelOption[];
  onOpenTrace: (trace: TraceEvent[]) => void;
  evaluationVerdict?: TestVerdict;
  onEvaluate?: (message: GenesisMessage, verdict: TestVerdict) => void;
  isEvaluating?: boolean;
  isSubstantialQuery?: boolean;
}) {
  const isAssistant = message.role === "assistant";
  const isThinking = message.content === "Pensando...";
  const isError = message.content === "Falha ao gerar resposta. Tente novamente.";
  const canEvaluate = isAssistant && !isThinking && !isError && (isSubstantialQuery ?? false);

  return (
    <article
      className={clsx(
        "w-full max-w-full rounded-[28px] border px-5 py-5 text-sm leading-relaxed shadow-[0_25px_70px_rgba(0,0,0,0.55)] transition-all sm:px-6 md:max-w-3xl",
        isAssistant
          ? isThinking
            ? "border-[rgba(176,111,255,0.22)] bg-[linear-gradient(180deg,rgba(24,23,35,0.98),rgba(16,18,27,0.98))] text-[#f6f7fb] shadow-[inset_0_1px_0_rgba(255,255,255,0.02),0_0_0_1px_rgba(139,61,255,0.06),0_25px_70px_rgba(19,10,38,0.34)]"
            : "border-[rgba(176,111,255,0.12)] bg-[linear-gradient(180deg,rgba(19,21,31,0.96),rgba(15,17,26,0.98))] text-[#f6f7fb] shadow-[inset_0_1px_0_rgba(255,255,255,0.015),0_25px_70px_rgba(0,0,0,0.5)]"
          : "border-[#8b3dff]/60 bg-[#8b3dff]/12 text-[#f6f0ff] ring-1 ring-[#8b3dff]/30 sm:ml-auto",
      )}
    >
      <div className="mb-2 flex items-center justify-between text-[10px] uppercase tracking-[0.35em] text-[#8f96ab]">
        <span>{isAssistant ? "DairyApp" : "Operador"}</span>
        <span className="text-[9px] tracking-[0.28em] text-[#545b72]">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>

      {isAssistant ? (
        <AssistantMessageBody content={message.content} isThinking={isThinking} trace={message.trace} />
      ) : (
        <p className="whitespace-pre-wrap break-words text-[15px] text-[#e6f4ff]" style={{ fontFamily: "var(--font-sans)" }}>
          {message.content}
        </p>
      )}

      <div className="mt-4 flex items-center gap-6 text-[10px] uppercase tracking-[0.35em] text-[#8f96ab]">
        {message.agentId ? (
          <span className="inline-flex items-center">
            Agente: {getAgentById(message.agentId)?.label ?? message.agentId}
          </span>
        ) : null}
        {message.modelId ? (
          <span className="inline-flex items-center">
            Modelo: {models.find((m) => m.id === message.modelId)?.label ?? message.modelId}
          </span>
        ) : null}
        {isAssistant && message.trace && message.trace.length > 0 && (
          <button
            type="button"
            onClick={() => onOpenTrace(message.trace!)}
            className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/[0.04] px-2.5 py-1 text-[10px] uppercase tracking-[0.28em] text-[#676e83] transition hover:border-[#8b3dff]/40 hover:text-[#b06fff]"
            title="Ver log da execução"
            aria-label="Ver log de execução"
          >
            <Activity className="h-3.5 w-3.5" />
            <span>Log</span>
          </button>
        )}
      </div>

      {canEvaluate && onEvaluate ? (
        <div className="mt-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-1 text-[10px] uppercase tracking-[0.35em] text-[#676e83]">Avaliação</span>
            <button
              type="button"
              onClick={() => onEvaluate(message, "correct")}
              disabled={isEvaluating}
              className={clsx(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[0.3em] transition",
                evaluationVerdict === "correct"
                  ? "border-emerald-400/45 bg-emerald-500/12 text-emerald-200"
                  : "border-white/12 bg-white/[0.03] text-[#8f96ab] hover:border-emerald-400/35 hover:text-emerald-200",
              )}
            >
              <ThumbsUp className="h-3.5 w-3.5" />
              <span>Correta</span>
            </button>
            <button
              type="button"
              onClick={() => onEvaluate(message, "partial")}
              disabled={isEvaluating}
              className={clsx(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[0.3em] transition",
                evaluationVerdict === "partial"
                  ? "border-amber-400/45 bg-amber-500/12 text-amber-200"
                  : "border-white/12 bg-white/[0.03] text-[#8f96ab] hover:border-amber-400/35 hover:text-amber-200",
              )}
            >
              <Minus className="h-3.5 w-3.5" />
              <span>Parcial</span>
            </button>
            <button
              type="button"
              onClick={() => onEvaluate(message, "incorrect")}
              disabled={isEvaluating}
              className={clsx(
                "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[0.3em] transition",
                evaluationVerdict === "incorrect"
                  ? "border-rose-400/45 bg-rose-500/12 text-rose-200"
                  : "border-white/12 bg-white/[0.03] text-[#8f96ab] hover:border-rose-400/35 hover:text-rose-200",
              )}
            >
              <ThumbsDown className="h-3.5 w-3.5" />
              <span>Incorreta</span>
            </button>
            {evaluationVerdict ? (
              <span className="ml-auto text-[10px] text-[#676e83]">Clique para editar</span>
            ) : null}
          </div>
        </div>
      ) : null}
    </article>
  );
});

const MessageList = memo(function MessageList({
  messages,
  models,
  onOpenTrace,
  evaluationsByMessageId,
  onEvaluate,
  isEvaluating,
}: {
  messages: GenesisMessage[];
  models: ModelOption[];
  onOpenTrace: (trace: TraceEvent[]) => void;
  evaluationsByMessageId: Record<string, { verdict: TestVerdict; comment?: string }>;
  onEvaluate: (message: GenesisMessage, verdict: TestVerdict) => void;
  isEvaluating: boolean;
}) {
  const substantialIds = useMemo(() => {
    const ids = new Set<string>();
    messages.forEach((msg) => {
      if (msg.role === "assistant" && isPrevUserSubstantial(messages, msg.id)) {
        ids.add(msg.id);
      }
    });
    return ids;
  }, [messages]);

  return (
    <>
      {messages.map((message) => (
        <MessageCard
          key={message.id}
          message={message}
          models={models}
          onOpenTrace={onOpenTrace}
          evaluationVerdict={evaluationsByMessageId[message.id]?.verdict}
          onEvaluate={onEvaluate}
          isEvaluating={isEvaluating}
          isSubstantialQuery={substantialIds.has(message.id)}
        />
      ))}
    </>
  );
});

const TestSessionSummary = memo(function TestSessionSummary({
  assistantMessageCount,
  evaluatedCount,
  correctCount,
  partialCount,
  incorrectCount,
  scorePercent,
  status,
  isSaving,
  onFinalize,
}: {
  assistantMessageCount: number;
  evaluatedCount: number;
  correctCount: number;
  partialCount: number;
  incorrectCount: number;
  scorePercent: number;
  status: "active" | "completed";
  isSaving: boolean;
  onFinalize: () => void;
}) {
  return (
    <section className="rounded-[24px] border border-[rgba(176,111,255,0.12)] bg-[linear-gradient(180deg,rgba(20,22,32,0.92),rgba(15,17,26,0.95))] px-5 py-4 shadow-[0_18px_44px_rgba(0,0,0,0.28)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-[0.38em] text-[#b06fff]/75">Sessão de teste</p>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-[#c9cdd6]">
            <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] uppercase tracking-[0.3em] text-white">
              <Check className="h-3.5 w-3.5 text-emerald-300" />
              <span>Score {scorePercent}%</span>
            </span>
            <span className="text-[#8f96ab]">
              {evaluatedCount} de {assistantMessageCount} respostas avaliadas
            </span>
            <span className="text-[#8f96ab]">· {correctCount} corretas</span>
            <span className="text-[#8f96ab]">· {partialCount} parciais</span>
            <span className="text-[#8f96ab]">· {incorrectCount} incorretas</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span
            className={clsx(
              "rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.35em]",
              status === "completed"
                ? "border-cyan-400/35 bg-cyan-500/10 text-cyan-200"
                : "border-amber-400/35 bg-amber-500/10 text-amber-200",
            )}
          >
            {status === "completed" ? "Sessão finalizada" : "Sessão ativa"}
          </span>
          {status === "active" ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onFinalize}
              disabled={isSaving || evaluatedCount === 0}
              className="border-[#8b3dff]/30 bg-[#8b3dff]/8 text-white hover:bg-[#8b3dff]/14 disabled:opacity-40"
              title={evaluatedCount === 0 ? "Avalie ao menos uma resposta antes de finalizar" : undefined}
            >
              {isSaving ? "Finalizando..." : "Finalizar sessão"}
            </Button>
          ) : null}
        </div>
      </div>
    </section>
  );
});

const FinalizationBanner = memo(function FinalizationBanner({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-2xl border border-emerald-500/30 bg-emerald-900/30 px-4 py-3 shadow-[0_8px_24px_rgba(0,0,0,0.3)]">
      <div className="flex items-center gap-3">
        <ClipboardCheck className="h-4 w-4 shrink-0 text-emerald-300" />
        <span className="text-sm text-emerald-200">
          Sessão finalizada com sucesso! Os resultados estão disponíveis em{" "}
          <Link href="/tests" className="font-semibold underline underline-offset-4 hover:text-white">
            /tests
          </Link>
          .
        </span>
      </div>
      <button
        type="button"
        onClick={onDismiss}
        className="shrink-0 rounded-full p-0.5 text-emerald-400/60 transition hover:text-emerald-200"
        aria-label="Fechar"
      >
        <svg width="13" height="13" viewBox="0 0 14 14" fill="none" aria-hidden>
          <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
});

const SessionComposer = memo(function SessionComposer({
  draft,
  isLoading,
  isSending,
  onChange,
  onSubmit,
  textareaRef,
}: {
  draft: string;
  isLoading: boolean;
  isSending: boolean;
  onChange: (value: string) => void;
  onSubmit: (event?: FormEvent<HTMLFormElement>) => void;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}) {
  return (
    <footer className="relative -mt-2 bg-transparent px-4 pb-5 pt-4 sm:px-6 lg:px-10">
      <form onSubmit={onSubmit} className="flex w-full flex-col gap-2 sm:flex-row sm:items-end sm:gap-4">
        <div className="flex-1">
          <label className="mb-1.5 block text-[11px] uppercase tracking-[0.35em] text-[#8a90a3]">
            Digite sua pergunta
          </label>
          <textarea
            ref={textareaRef}
            value={draft}
            onChange={(e) => onChange(e.target.value)}
            placeholder={isLoading ? "Carregando..." : "Ex: Qual o limite de coliformes para queijo minas frescal?"}
            className="h-28 w-full resize-none rounded-[22px] border border-[rgba(176,111,255,0.1)] bg-[linear-gradient(180deg,rgba(22,21,31,0.88),rgba(17,18,27,0.92))] px-5 py-4 text-sm text-white placeholder:text-[#777d90] shadow-[inset_0_1px_0_rgba(255,255,255,0.01),0_12px_28px_rgba(0,0,0,0.14)] focus:border-[rgba(176,111,255,0.42)] focus:outline-none focus:ring-[3px] focus:ring-[rgba(139,61,255,0.14)] sm:h-32 lg:h-[110px]"
            disabled={isLoading}
          />
        </div>
        <div className="flex w-full items-end sm:w-auto sm:self-stretch sm:pt-6">
          <Button
            type="submit"
            disabled={isLoading || isSending || !draft.trim()}
            className="h-12 w-full rounded-[18px] px-6 text-[11px] shadow-[0_20px_40px_rgba(43,20,79,0.36)] sm:min-h-[110px] sm:w-auto sm:self-stretch sm:px-10"
          >
            Enviar
          </Button>
        </div>
      </form>
    </footer>
  );
});

const AuthScreen = memo(function AuthScreen({
  isLoggingIn,
  authMessage,
  passkeyInput,
  onPasskeyChange,
  onSubmit,
}: {
  isLoggingIn: boolean;
  authMessage: string | null;
  passkeyInput: string;
  onPasskeyChange: (value: string) => void;
  onSubmit: (event?: FormEvent<HTMLFormElement>) => void;
}) {
  return (
    <div className="relative min-h-screen overflow-hidden bg-[var(--cmdx-bg)] px-4 py-10 sm:px-6 lg:px-10">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_16%,rgba(176,111,255,0.12),transparent_26%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_22%_84%,rgba(139,61,255,0.08),transparent_34%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.018),transparent_22%)]" />

      <div className="relative mx-auto flex max-w-4xl flex-col items-center gap-8 pt-4 text-center">
        <div className="flex flex-col items-center gap-5">
          <div className="flex items-center justify-center rounded-[30px] border border-[rgba(176,111,255,0.16)] bg-[linear-gradient(180deg,rgba(23,26,34,0.94),rgba(14,16,23,0.96))] p-2.5 shadow-[0_24px_70px_rgba(43,20,79,0.34)] ring-1 ring-[rgba(176,111,255,0.08)]">
            <Image
              src="/commandix-logo.png"
              alt="Commandix"
              width={140}
              height={140}
              priority
              className="h-28 w-28 rounded-2xl drop-shadow-[0_18px_54px_rgba(139,61,255,0.3)]"
              style={{ objectFit: "contain" }}
            />
          </div>
          <div className="space-y-2">
            <h1
              className="text-4xl font-black uppercase leading-tight text-white sm:text-5xl"
              style={{ fontFamily: "var(--font-condensed)" }}
            >
              Commandix AI
            </h1>
            <p className="text-[11px] uppercase tracking-[0.38em] text-[#8f96ab]">Ambiente teste de agentes</p>
          </div>
        </div>

        <section className="w-full max-w-xl text-left">
          <div className="relative overflow-hidden rounded-[32px] border border-[rgba(176,111,255,0.14)] bg-[linear-gradient(180deg,rgba(23,26,34,0.96),rgba(11,13,20,0.98))] p-6 text-[#c9cdd6] shadow-[0_40px_100px_rgba(0,0,0,0.75)] ring-1 ring-[rgba(176,111,255,0.06)] sm:p-8">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(176,111,255,0.12),transparent_42%)]" />
            <div className="relative flex items-center justify-between">
              <p className="text-[11px] uppercase tracking-[0.5em] text-[#b06fff]/80">Acesso restrito</p>
              <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.025] px-3 py-1 text-[11px] uppercase tracking-[0.35em] text-[#8f96ab]">
                <span className="h-2 w-2 rounded-full bg-[#58d38f]" aria-hidden />
                <Link
                  href="https://commandix.tech/"
                  target="_blank"
                  rel="noreferrer"
                  className="transition hover:text-white"
                >
                  Commandix
                </Link>
              </span>
            </div>
            <h2
              className="relative mt-3 text-3xl font-bold uppercase text-white sm:text-4xl"
              style={{ fontFamily: "var(--font-condensed)" }}
            >
              DairyApp
            </h2>
            <p className="relative mt-3 text-sm text-[#8f96ab]">
              Insira a passkey gerada. Ao liberar, você cai direto no console em tempo real.
            </p>
            <form onSubmit={onSubmit} className="relative mt-6 space-y-4 sm:mt-8">
              <label className="block text-[11px] uppercase tracking-[0.3em] text-[#8f96ab]">Senha de acesso</label>
              <input
                type="password"
                value={passkeyInput}
                onChange={(e) => onPasskeyChange(e.target.value)}
                placeholder="********"
                className="w-full rounded-2xl border border-[rgba(176,111,255,0.24)] bg-[linear-gradient(180deg,rgba(34,37,49,0.94),rgba(28,31,42,0.96))] px-4 py-3 text-sm text-white placeholder:text-[#7b8298] shadow-[inset_0_1px_0_rgba(255,255,255,0.016),0_18px_42px_rgba(0,0,0,0.18)] focus:border-[rgba(176,111,255,0.62)] focus:outline-none focus:ring-2 focus:ring-[rgba(139,61,255,0.16)]"
              />
              {authMessage ? (
                <Alert variant="error" className="mt-3">
                  <span className="text-sm">{authMessage}</span>
                </Alert>
              ) : null}
              <Button
                type="submit"
                disabled={isLoggingIn || !passkeyInput.trim()}
                className="w-full justify-center py-4 text-sm shadow-[0_20px_40px_rgba(43,20,79,0.34)]"
              >
                {isLoggingIn ? "Validando..." : "Liberar console"}
              </Button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
});

const EmptySessionState = memo(function EmptySessionState({
  isLoading,
  onCreateSession,
}: {
  isLoading: boolean;
  onCreateSession: () => void;
}) {
  return (
    <div className="flex h-full items-center justify-center">
      <Card className="relative w-full max-w-3xl overflow-hidden border border-[rgba(176,111,255,0.16)] bg-[linear-gradient(180deg,rgba(23,26,34,0.95),rgba(15,17,25,0.97))] text-white shadow-[0_45px_120px_rgba(0,0,0,0.65)] ring-1 ring-[rgba(176,111,255,0.06)]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(176,111,255,0.12),transparent_42%)]" />
        <div className="pointer-events-none absolute inset-x-10 bottom-0 h-24 bg-[radial-gradient(circle_at_center,rgba(139,61,255,0.14),transparent_70%)] opacity-70 blur-2xl" />
        <CardHeader className="relative text-center">
          <p className="text-[11px] uppercase tracking-[0.4em] text-[#b06fff]/80">Seleção de Sessão</p>
          <h2 className="text-3xl font-bold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
            Iniciar Consulta
          </h2>
        </CardHeader>
        <CardContent className="relative">
          <div className="grid gap-4 sm:grid-cols-2">
            <button
              type="button"
              onClick={onCreateSession}
              disabled={isLoading}
              className={clsx(
                "flex h-44 flex-col justify-between rounded-3xl border px-5 py-4 text-left transition-all",
                "border-[#8b3dff]/60 bg-[linear-gradient(180deg,rgba(139,61,255,0.16),rgba(43,20,79,0.2))] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.03),0_30px_70px_rgba(43,20,79,0.38)] hover:border-[#b06fff]/75 hover:translate-y-[-1px] disabled:opacity-50",
              )}
            >
              <span className="text-xl font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                Nova Sessão
              </span>
              <span className="text-xs text-[#d7c9ee]">Inicie uma nova conversa com um agente especializado em laticínios.</span>
            </button>
            <Link
              href="/history"
              className="flex h-44 flex-col justify-between rounded-3xl border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01))] px-5 py-4 text-left text-[#c9cdd6] shadow-[inset_0_1px_0_rgba(255,255,255,0.02)] transition hover:border-[#8b3dff]/28 hover:bg-[rgba(176,111,255,0.04)] hover:translate-y-[-1px]"
            >
              <span className="text-xl font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                Ver Histórico
              </span>
              <span className="text-xs text-[#8f96ab]">Retome ou revise consultas realizadas anteriormente.</span>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
});

export function ChatPane() {
  const { models } = useGenesisCatalog();
  const { isLoading, isSending, currentSessionId, messagesBySession, sendMessage, createSession } =
    useGenesisConversation();
  const { token, isReady: authReady, isLoggingIn, loginError, login } = useAuth();
  const messages = useMemo(
    () => (currentSessionId ? (messagesBySession[currentSessionId] ?? []) : []),
    [messagesBySession, currentSessionId],
  );
  const hasActiveSession = Boolean(currentSessionId);
  const [draft, setDraft] = useState("");
  const [passkeyInput, setPasskeyInput] = useState("");
  const [localLoginError, setLocalLoginError] = useState<string | null>(null);
  const [activeTrace, setActiveTrace] = useState<TraceEvent[] | null>(null);

  // Evaluation modal state
  const [evaluationModal, setEvaluationModal] = useState<{
    message: GenesisMessage;
    initialVerdict: TestVerdict;
    initialComment?: string;
    initialErrorCategory?: TestErrorCategory;
    initialExpectedAnswer?: string;
  } | null>(null);

  // Finalization success banner
  const [showFinalizationBanner, setShowFinalizationBanner] = useState(false);

  const {
    session: testSession,
    evaluationsByMessageId,
    isSaving: isSavingEvaluation,
    errorMessage: threadTestingError,
    assistantMessageCount,
    saveEvaluation,
    finalizeSession,
    clearErrorMessage,
  } = useThreadTesting(currentSessionId || null, messages);

  // Auto-dismiss error after 6s
  useEffect(() => {
    if (!threadTestingError) return;
    const t = setTimeout(clearErrorMessage, 6000);
    return () => clearTimeout(t);
  }, [threadTestingError, clearErrorMessage]);

  // Auto-dismiss finalization banner after 7s
  useEffect(() => {
    if (!showFinalizationBanner) return;
    const t = setTimeout(() => setShowFinalizationBanner(false), 7000);
    return () => clearTimeout(t);
  }, [showFinalizationBanner]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const [userHasScrolled, setUserHasScrolled] = useState(false);
  const lastMessageCountRef = useRef(0);

  const handleSubmit = useCallback(
    async (event?: FormEvent<HTMLFormElement>) => {
      event?.preventDefault();
      const trimmed = draft.trim();
      if (!trimmed || isLoading || isSending || !token || !hasActiveSession) return;
      setDraft("");
      setUserHasScrolled(false);
      try {
        await sendMessage(trimmed);
      } catch {
        setDraft(trimmed);
      }
    },
    [draft, hasActiveSession, isLoading, isSending, token, sendMessage],
  );

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
      setLocalLoginError(error instanceof Error ? error.message : "Falha ao autenticar");
    }
  }

  const handleOpenEvaluationModal = useCallback(
    (message: GenesisMessage, verdict: TestVerdict) => {
      const existing = evaluationsByMessageId[message.id];
      setEvaluationModal({
        message,
        initialVerdict: verdict,
        initialComment: existing?.comment,
        initialErrorCategory: existing?.error_category,
        initialExpectedAnswer: existing?.expected_answer,
      });
    },
    [evaluationsByMessageId],
  );

  const handleModalSave = useCallback(
    async (
      verdict: TestVerdict,
      payload: { comment: string; errorCategory?: TestErrorCategory; expectedAnswer?: string },
    ) => {
      if (!evaluationModal) return;
      await saveEvaluation(evaluationModal.message, verdict, {
        comment: payload.comment || undefined,
        errorCategory: payload.errorCategory,
        expectedAnswer: payload.expectedAnswer,
      });
    },
    [evaluationModal, saveEvaluation],
  );

  const handleFinalize = useCallback(async () => {
    const success = await finalizeSession();
    if (success) {
      setShowFinalizationBanner(true);
    }
  }, [finalizeSession]);

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

  useEffect(() => {
    const mainElement = mainRef.current;
    if (!mainElement) return;
    function handleUserScroll(event: Event) {
      const target = event.target as HTMLElement;
      if (!target) return;
      const { scrollTop, scrollHeight, clientHeight } = target;
      if (scrollHeight - scrollTop - clientHeight >= 100) setUserHasScrolled(true);
    }
    mainElement.addEventListener("scroll", handleUserScroll, { passive: true });
    mainElement.addEventListener("wheel", handleUserScroll, { passive: true });
    mainElement.addEventListener("touchmove", handleUserScroll, { passive: true });
    return () => {
      mainElement.removeEventListener("scroll", handleUserScroll);
      mainElement.removeEventListener("wheel", handleUserScroll);
      mainElement.removeEventListener("touchmove", handleUserScroll);
    };
  }, []);

  useEffect(() => {
    const hasNewMessages = messages.length > lastMessageCountRef.current;
    lastMessageCountRef.current = messages.length;
    if (!userHasScrolled || hasNewMessages) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, userHasScrolled]);

  useEffect(() => {
    setUserHasScrolled(false);
    lastMessageCountRef.current = 0;
  }, [currentSessionId]);

  if (!authReady) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--cmdx-bg)] px-4">
        <p className="text-xs uppercase tracking-[0.45em] text-[#8f96ab]">Sincronizando credenciais...</p>
      </div>
    );
  }

  if (!token) {
    return (
      <AuthScreen
        isLoggingIn={isLoggingIn}
        authMessage={localLoginError || loginError}
        passkeyInput={passkeyInput}
        onPasskeyChange={setPasskeyInput}
        onSubmit={handleLoginSubmit}
      />
    );
  }

  return (
    <div className="relative flex h-full flex-col overflow-hidden bg-[var(--cmdx-bg)]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_14%,rgba(139,61,255,0.05),transparent_32%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.018),transparent_24%)]" />
      <main
        ref={mainRef}
        className="relative flex flex-1 flex-col gap-5 overflow-y-auto px-4 py-6 text-[#c9cdd6] sm:px-6 lg:px-10"
      >
        {assistantMessageCount > 0 ? (
          <TestSessionSummary
            assistantMessageCount={assistantMessageCount}
            evaluatedCount={testSession?.metrics.evaluated_count ?? 0}
            correctCount={testSession?.metrics.correct_count ?? 0}
            partialCount={testSession?.metrics.partial_count ?? 0}
            incorrectCount={testSession?.metrics.incorrect_count ?? 0}
            scorePercent={testSession?.metrics.score_percent ?? 0}
            status={testSession?.status ?? "active"}
            isSaving={isSavingEvaluation}
            onFinalize={() => {
              void handleFinalize();
            }}
          />
        ) : null}

        {showFinalizationBanner ? (
          <FinalizationBanner onDismiss={() => setShowFinalizationBanner(false)} />
        ) : null}

        {threadTestingError ? (
          <Alert variant="error" onDismiss={clearErrorMessage}>
            <span className="text-sm">{threadTestingError}</span>
          </Alert>
        ) : null}

        <div className="relative flex-1 space-y-4">
          {!hasActiveSession ? (
            <EmptySessionState isLoading={isLoading} onCreateSession={handleCreateSession} />
          ) : isLoading ? (
            <Card className="border border-white/12 bg-white/5 text-center text-[#8f96ab]">
              <CardHeader>Carregando sessões</CardHeader>
              <CardContent>Aguarde enquanto conectamos ao LangGraph.</CardContent>
            </Card>
          ) : messages.length === 0 ? (
            <div className="flex h-full items-center justify-center">
              <p
                className="select-none text-[2.6rem] font-black uppercase tracking-[0.22em] text-[rgba(245,246,248,0.028)]"
                style={{ fontFamily: "var(--font-condensed)" }}
              >
                Dairy AI
              </p>
            </div>
          ) : (
            <MessageList
              messages={messages}
              models={models}
              onOpenTrace={setActiveTrace}
              evaluationsByMessageId={evaluationsByMessageId}
              onEvaluate={handleOpenEvaluationModal}
              isEvaluating={isSavingEvaluation}
            />
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {activeTrace && <LazyTraceModal trace={activeTrace} onClose={() => setActiveTrace(null)} />}

      {evaluationModal ? (
        <LazyEvaluationModal
          message={evaluationModal.message}
          initialVerdict={evaluationModal.initialVerdict}
          initialComment={evaluationModal.initialComment}
          initialErrorCategory={evaluationModal.initialErrorCategory}
          initialExpectedAnswer={evaluationModal.initialExpectedAnswer}
          isSaving={isSavingEvaluation}
          onSave={handleModalSave}
          onClose={() => setEvaluationModal(null)}
        />
      ) : null}

      {hasActiveSession ? (
        <SessionComposer
          draft={draft}
          isLoading={isLoading}
          isSending={isSending}
          onChange={setDraft}
          onSubmit={handleSubmit}
          textareaRef={textareaRef}
        />
      ) : null}
    </div>
  );
}
