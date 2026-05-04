"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { CheckCircle2, CircleDashed, ClipboardList, ExternalLink, RefreshCw, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/state/useAuth";
import { useGenesisConversation } from "@/state/useGenesisUI";

interface TestSessionMetrics {
  evaluated_count: number;
  correct_count: number;
  partial_count: number;
  incorrect_count: number;
  score_percent: number;
}

interface TestSessionItem {
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

type FilterTab = "all" | "active" | "completed";

export function TestSessionsView() {
  const { token } = useAuth();
  const router = useRouter();
  const { selectSession } = useGenesisConversation();
  const [sessions, setSessions] = useState<TestSessionItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [filter, setFilter] = useState<FilterTab>("all");
  const [navigatingId, setNavigatingId] = useState<string | null>(null);

  const load = useCallback(
    async (silent = false) => {
      if (!token) return;
      if (silent) setIsRefreshing(true);
      else setIsLoading(true);
      try {
        const response = await fetch("/api/tests/sessions", {
          headers: { Authorization: `Bearer ${token}` },
          cache: "no-store",
        });
        if (!response.ok) return;
        const data = (await response.json()) as { sessions?: TestSessionItem[] };
        setSessions(data.sessions ?? []);
      } finally {
        setIsLoading(false);
        setIsRefreshing(false);
      }
    },
    [token],
  );

  useEffect(() => {
    load().catch(console.error);
  }, [load]);

  const handleOpenThread = useCallback(
    async (session: TestSessionItem) => {
      if (navigatingId) return;
      setNavigatingId(session.id);
      try {
        await selectSession(session.thread_id);
        router.push("/");
      } catch {
        setNavigatingId(null);
      }
    },
    [navigatingId, selectSession, router],
  );

  const summary = useMemo(() => {
    const total = sessions.length;
    const completed = sessions.filter((s) => s.status === "completed").length;
    const active = sessions.filter((s) => s.status === "active").length;
    const avgScore =
      total > 0
        ? Math.round(sessions.reduce((acc, s) => acc + s.metrics.score_percent, 0) / total)
        : 0;
    return { total, completed, active, avgScore };
  }, [sessions]);

  const topSessions = useMemo(
    () => [...sessions].sort((a, b) => b.metrics.score_percent - a.metrics.score_percent).slice(0, 3),
    [sessions],
  );

  const filtered = useMemo(() => {
    if (filter === "active") return sessions.filter((s) => s.status === "active");
    if (filter === "completed") return sessions.filter((s) => s.status === "completed");
    return sessions;
  }, [sessions, filter]);

  return (
    <div className="flex min-h-screen flex-1 flex-col bg-[var(--cmdx-bg)] px-4 py-8 text-[#c9cdd6] sm:px-6 lg:px-10 lg:py-12">
      {/* Header */}
      <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between lg:mb-10">
        <div>
          <div className="text-[11px] uppercase tracking-[0.4em] text-[#b06fff]/80">Avaliação de Agentes</div>
          <h1 className="text-3xl font-bold uppercase text-white" style={{ fontFamily: "var(--font-condensed)" }}>
            Sessões de Teste
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-[#8f96ab]">
            Avaliações feitas no console — score por sessão, volume avaliado e sinais rápidos de qualidade.
          </p>
        </div>
        <div className="flex items-center gap-3 self-start sm:self-auto">
          <button
            type="button"
            onClick={() => load(true).catch(console.error)}
            disabled={isRefreshing || isLoading}
            className="inline-flex items-center gap-2 rounded-full border border-white/12 bg-white/[0.03] px-3 py-2 text-[10px] uppercase tracking-[0.3em] text-[#8f96ab] transition hover:border-[#8b3dff]/30 hover:text-[#b06fff] disabled:opacity-40"
            title="Atualizar lista"
          >
            <RefreshCw className={clsx("h-3.5 w-3.5", isRefreshing && "animate-spin")} />
            <span>Atualizar</span>
          </button>
          <Link href="/">
            <Button>Ir para o console</Button>
          </Link>
        </div>
      </header>

      {/* Metric cards */}
      <section className="mb-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Sessões" value={summary.total} icon={<ClipboardList className="h-4 w-4" />} />
        <MetricCard label="Ativas" value={summary.active} icon={<CircleDashed className="h-4 w-4" />} />
        <MetricCard label="Finalizadas" value={summary.completed} icon={<CheckCircle2 className="h-4 w-4" />} />
        <MetricCard label="Score médio" value={`${summary.avgScore}%`} icon={<CheckCircle2 className="h-4 w-4" />} accent />
      </section>

      {/* Top sessions */}
      {topSessions.length > 0 ? (
        <section className="mb-8 rounded-[28px] border border-[rgba(176,111,255,0.14)] bg-[linear-gradient(180deg,rgba(23,26,34,0.9),rgba(14,16,24,0.95))] px-5 py-5 shadow-[0_24px_70px_rgba(0,0,0,0.32)]">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-[11px] uppercase tracking-[0.38em] text-[#b06fff]/75">Resumo rápido</p>
              <h2 className="mt-1 text-xl font-semibold text-white">Melhores resultados recentes</h2>
            </div>
            <p className="text-sm text-[#8f96ab]">Visão rápida para reunião e acompanhamento de qualidade.</p>
          </div>
          <div className="mt-4 grid gap-3 lg:grid-cols-3">
            {topSessions.map((session, index) => (
              <button
                key={session.id}
                type="button"
                onClick={() => void handleOpenThread(session)}
                disabled={Boolean(navigatingId)}
                className="group rounded-[22px] border border-white/10 bg-white/[0.025] px-4 py-4 text-left transition hover:border-[#8b3dff]/35 hover:bg-white/[0.04] disabled:opacity-60"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[10px] uppercase tracking-[0.35em] text-[#8f96ab]">Top {index + 1}</span>
                  <span className={clsx("rounded-full px-2.5 py-1 text-[10px] uppercase tracking-[0.3em]", getScoreTone(session.metrics.score_percent))}>
                    {session.metrics.score_percent}%
                  </span>
                </div>
                <p className="mt-3 line-clamp-2 text-base font-semibold text-white">{session.title}</p>
                <ScoreBar value={session.metrics.score_percent} className="mt-3" />
                <p className="mt-2 text-sm text-[#8f96ab]">
                  {session.metrics.correct_count} corretas · {session.metrics.partial_count} parciais · {session.metrics.incorrect_count} incorretas
                </p>
                <p className="mt-2 flex items-center gap-1 text-[10px] uppercase tracking-[0.3em] text-[#8b3dff]/60 opacity-0 transition group-hover:opacity-100">
                  <ExternalLink className="h-3 w-3" />
                  <span>Abrir no console</span>
                </p>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {/* Filter tabs */}
      {sessions.length > 0 ? (
        <div className="mb-5 flex items-center gap-2">
          {(["all", "active", "completed"] as FilterTab[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setFilter(tab)}
              className={clsx(
                "rounded-full border px-3 py-1.5 text-[10px] uppercase tracking-[0.32em] transition",
                filter === tab
                  ? "border-[#8b3dff]/50 bg-[#8b3dff]/12 text-[#b06fff]"
                  : "border-white/10 bg-white/[0.02] text-[#8f96ab] hover:border-white/20 hover:text-white",
              )}
            >
              {tab === "all" ? `Todas (${sessions.length})` : tab === "active" ? `Ativas (${summary.active})` : `Finalizadas (${summary.completed})`}
            </button>
          ))}
        </div>
      ) : null}

      {/* Session grid */}
      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-64 animate-pulse rounded-3xl border border-white/10 bg-white/5" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <div className="mt-12 flex flex-col items-center gap-6 rounded-3xl border border-dashed border-white/12 bg-white/5 px-8 py-16 text-center">
          <p className="text-sm text-[#8f96ab]">
            {sessions.length === 0
              ? "Nenhuma sessão de teste registrada ainda."
              : "Nenhuma sessão neste filtro."}
          </p>
          {sessions.length === 0 ? (
            <Link href="/">
              <Button>Começar a testar</Button>
            </Link>
          ) : (
            <button
              type="button"
              onClick={() => setFilter("all")}
              className="text-sm text-[#b06fff] underline underline-offset-4 hover:text-white"
            >
              Ver todas as sessões
            </button>
          )}
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((session) => (
            <article
              key={session.id}
              className="group flex flex-col gap-4 rounded-3xl border border-white/15 bg-[rgba(23,26,34,0.92)] px-5 py-5 shadow-[0_25px_60px_rgba(0,0,0,0.34)] transition hover:border-[#8b3dff]/30"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-[10px] uppercase tracking-[0.35em] text-[#8f96ab]">Thread avaliada</p>
                  <h2 className="mt-1 truncate text-xl font-semibold uppercase text-white" style={{ fontFamily: "var(--font-condensed)" }}>
                    {session.title}
                  </h2>
                </div>
                <span
                  className={clsx(
                    "shrink-0 rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.35em]",
                    session.status === "completed"
                      ? "border-cyan-400/35 bg-cyan-500/10 text-cyan-200"
                      : "border-amber-400/35 bg-amber-500/10 text-amber-200",
                  )}
                >
                  {session.status === "completed" ? "Finalizada" : "Ativa"}
                </span>
              </div>

              {/* Score */}
              <div className="rounded-2xl border border-white/8 bg-white/[0.025] px-4 py-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-[0.3em] text-[#8f96ab]">Score</span>
                  <span className={clsx("text-lg font-bold", getScoreTextTone(session.metrics.score_percent))}>
                    {session.metrics.score_percent}%
                  </span>
                </div>
                <ScoreBar value={session.metrics.score_percent} />
              </div>

              {/* Mini stats */}
              <div className="grid grid-cols-2 gap-2">
                <MiniStat label="Avaliadas" value={session.metrics.evaluated_count} />
                <MiniStat label="Corretas" value={session.metrics.correct_count} tone="text-emerald-300" />
                <MiniStat label="Parciais" value={session.metrics.partial_count} tone="text-amber-300" />
                <MiniStat
                  label="Incorretas"
                  value={session.metrics.incorrect_count}
                  tone={session.metrics.incorrect_count > 0 ? "text-rose-300" : undefined}
                />
              </div>

              {/* Footer row */}
              <div className="mt-auto flex items-center justify-between gap-3">
                <span className="text-[10px] uppercase tracking-[0.3em] text-[#676e83]">
                  {new Date(session.started_at).toLocaleString("pt-BR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" })}
                </span>
                {session.metrics.incorrect_count > 0 ? (
                  <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-[0.3em] text-rose-300">
                    <XCircle className="h-3.5 w-3.5" />
                    <span>Atenção</span>
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[10px] uppercase tracking-[0.3em] text-emerald-300">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Sem erros</span>
                  </span>
                )}
              </div>

              {/* Open in console button */}
              <button
                type="button"
                onClick={() => void handleOpenThread(session)}
                disabled={Boolean(navigatingId)}
                className="flex w-full items-center justify-center gap-2 rounded-2xl border border-[#8b3dff]/20 bg-[#8b3dff]/6 py-2.5 text-[10px] uppercase tracking-[0.32em] text-[#8b3dff]/70 transition hover:border-[#8b3dff]/40 hover:bg-[#8b3dff]/10 hover:text-[#b06fff] disabled:opacity-40"
              >
                {navigatingId === session.id ? (
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <ExternalLink className="h-3.5 w-3.5" />
                )}
                <span>{navigatingId === session.id ? "Abrindo..." : "Abrir no console"}</span>
              </button>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

function ScoreBar({ value, className }: { value: number; className?: string }) {
  return (
    <div className={clsx("h-1.5 w-full overflow-hidden rounded-full bg-white/8", className)}>
      <div
        className={clsx(
          "h-full rounded-full transition-all",
          value >= 80 ? "bg-emerald-400" : value >= 50 ? "bg-amber-400" : "bg-rose-400",
        )}
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

function MetricCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: string | number;
  icon: ReactNode;
  accent?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-3xl border px-5 py-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]",
        accent
          ? "border-[rgba(176,111,255,0.2)] bg-[linear-gradient(180deg,rgba(139,61,255,0.08),rgba(43,20,79,0.06))]"
          : "border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))]",
      )}
    >
      <div className="flex items-center justify-between text-[#8f96ab]">
        <span className="text-[10px] uppercase tracking-[0.35em]">{label}</span>
        {icon}
      </div>
      <div
        className={clsx("mt-3 text-3xl font-bold", accent ? "text-[#b06fff]" : "text-white")}
        style={{ fontFamily: "var(--font-condensed)" }}
      >
        {value}
      </div>
    </div>
  );
}

function MiniStat({ label, value, tone }: { label: string; value: string | number; tone?: string }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.3em] text-[#8f96ab]">{label}</p>
      <p className={clsx("mt-1 text-base font-semibold text-white", tone)}>{value}</p>
    </div>
  );
}

function getScoreTone(score: number) {
  if (score >= 80) return "bg-emerald-500/12 text-emerald-200";
  if (score >= 50) return "bg-amber-500/12 text-amber-200";
  return "bg-rose-500/12 text-rose-200";
}

function getScoreTextTone(score: number) {
  if (score >= 80) return "text-emerald-300";
  if (score >= 50) return "text-amber-300";
  return "text-rose-300";
}
