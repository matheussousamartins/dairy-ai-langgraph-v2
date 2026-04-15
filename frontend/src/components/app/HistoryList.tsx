"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useRouter } from "next/navigation";
import { useGenesisUI } from "@/state/useGenesisUI";

export function HistoryList() {
  const router = useRouter();
  const { isLoading, sessions, messagesBySession, selectSession } = useGenesisUI();

  const items = useMemo(
    () =>
      sessions.map((session) => {
        const msgs = messagesBySession[session.id] ?? [];
        const firstUserMsg = msgs.find((m) => m.role === "user")?.content ?? "";
        const lastAgentMsg = msgs.filter((m) => m.role === "assistant").slice(-1)[0];
        const agentId = lastAgentMsg?.modelId ?? null;
        return {
          ...session,
          messageCount: msgs.length,
          question: firstUserMsg,
          agentId,
        };
      }),
    [sessions, messagesBySession],
  );

  return (
    <div className="flex min-h-screen flex-1 flex-col bg-gradient-to-br from-[#05080f] via-[#0b1428] to-[#080d18] px-4 py-8 text-[#dfdecf] sm:px-6 lg:px-10 lg:py-12">
      <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between lg:mb-10">
        <div>
          <div className="text-[11px] uppercase tracking-[0.4em] text-[#05adca]/70">Histórico de Consultas</div>
          <h1
            className="text-3xl font-bold uppercase text-white"
            style={{ fontFamily: "var(--font-condensed)" }}
          >
            Consultas Recentes
          </h1>
          <p className="mt-2 max-w-xl text-sm text-[#7f8baf]">
            Todas as suas consultas ficam salvas. Clique em qualquer uma para retomar a conversa.
          </p>
        </div>
        <Link
          href="/"
          className="inline-flex items-center justify-center self-start rounded-full border border-[#1086ad]/60 bg-[#1086ad]/10 px-5 py-2.5 text-[11px] uppercase tracking-[0.4em] text-white transition hover:bg-[#1086ad]/20 sm:self-auto"
        >
          Nova Consulta
        </Link>
      </header>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-44 animate-pulse rounded-3xl border border-white/10 bg-white/5"
            />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="mt-12 flex flex-col items-center gap-6 rounded-3xl border border-dashed border-white/12 bg-white/5 px-8 py-16 text-center">
          <p className="text-sm text-[#7f8baf]">Nenhuma consulta realizada ainda.</p>
          <Link
            href="/"
            className="rounded-full border border-[#1086ad]/60 bg-[#1086ad]/10 px-6 py-2.5 text-[11px] uppercase tracking-[0.4em] text-white transition hover:bg-[#1086ad]/20"
          >
            Iniciar primeira consulta
          </Link>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {items.map((item) => (
            <button
              key={item.id}
              type="button"
              className="group flex flex-col gap-3 rounded-3xl border border-white/15 bg-[rgba(9,14,26,0.9)] px-5 py-5 text-left transition-all hover:border-[#1086ad]/50 hover:bg-[#1086ad]/5 hover:shadow-[0_35px_80px_rgba(0,0,0,0.55)]"
              onClick={() => {
                selectSession(item.id)
                  .catch(console.error)
                  .finally(() => router.push("/"));
              }}
            >
              <div className="flex items-start justify-between gap-2">
                <span
                  className="text-base font-semibold uppercase text-white"
                  style={{ fontFamily: "var(--font-condensed)" }}
                >
                  {item.title}
                </span>
                {item.agentId && (
                  <span className="shrink-0 rounded-full border border-white/15 bg-white/5 px-2.5 py-0.5 text-[10px] uppercase tracking-[0.3em] text-[#9ba3c0]">
                    {item.agentId}
                  </span>
                )}
              </div>

              {item.question && (
                <p className="line-clamp-3 text-sm text-[#9ba3c0]">
                  {item.question}
                </p>
              )}

              <div className="mt-auto flex items-center justify-between text-[10px] uppercase tracking-[0.35em] text-[#5c6383]">
                <span>{item.messageCount} mensagens</span>
                <span>{new Date(item.createdAt).toLocaleString("pt-BR")}</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
