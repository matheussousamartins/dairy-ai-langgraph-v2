"use client";

import Link from "next/link";
import { useMemo } from "react";
import clsx from "clsx";
import { useGenesisUI } from "@/state/useGenesisUI";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/state/useAuth";

interface SidebarProps {
  isMobile?: boolean;
}

export function Sidebar({ isMobile = false }: SidebarProps) {
  const { isLoading, sessions, currentSessionId, selectSession, createSession, messagesBySession } = useGenesisUI();
  const { token } = useAuth();
  const hasAccess = Boolean(token);

  const recentSessions = useMemo(
    () => [...sessions].sort((a, b) => b.createdAt - a.createdAt).slice(0, 2),
    [sessions],
  );

  return (
    <aside
      className={clsx(
        "relative flex w-80 flex-col gap-6 border-white/10 bg-gradient-to-b from-[#111a32]/90 via-[#0c1324]/92 to-[#080f1b]/95 p-7 text-[#dfdecf] shadow-[0_30px_80px_rgba(0,0,0,0.65)] backdrop-blur-2xl",
        isMobile
          ? "w-full max-w-[22rem] rounded-2xl border border-white/15 max-h-[calc(100vh-1.5rem)] overflow-y-auto"
          : "sticky top-0 h-screen border-r",
      )}
    >
      <section className="space-y-3">
        <div>
          <p className="text-[10px] uppercase tracking-[0.35em] text-[#05adca]/70">Ambiente de Testes de Agentes</p>
          <h2 className="text-2xl font-black uppercase text-white" style={{ fontFamily: "var(--font-condensed)" }}>
            Commandix Tech
          </h2>
          <p className="mt-1 text-xs uppercase tracking-[0.3em] text-[#7f8baf]">Threads Recentes</p>
        </div>
        <Button
          onClick={() => createSession().catch(console.error)}
          disabled={isLoading || !hasAccess}
          className="w-full justify-center border border-white/20 bg-white/5 text-sm uppercase tracking-[0.35em] text-white hover:border-[#1086ad] hover:bg-[#1086ad]/15 disabled:opacity-40"
        >
          Nova Sessão
        </Button>
      </section>

      <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <section className="flex-1 overflow-hidden">
        <div className="-mr-3 flex h-full flex-col gap-2 overflow-y-auto pr-3">
          {isLoading ? (
            <div className="space-y-2 rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-[#7f8baf]">
              Carregando threads…
            </div>
          ) : recentSessions.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/12 bg-white/5 p-6 text-center text-xs text-[#7f8baf]">
              Nenhuma thread ativa ainda.
            </div>
          ) : (
            recentSessions.map((session) => {
              const active = session.id === currentSessionId;
              const sessionMessages = messagesBySession[session.id] ?? [];
              const lastMessage = sessionMessages.slice(-1)[0]?.content ?? "";
              const createdLabel = new Date(session.createdAt).toLocaleDateString("pt-BR");
              return (
                <button
                  key={session.id}
                  onClick={() => selectSession(session.id).catch(console.error)}
                  className={clsx(
                    "flex flex-col gap-1 rounded-2xl border px-4 py-3 text-left transition-all",
                    active
                      ? "border-[#1086ad]/70 bg-[#1086ad]/12 text-white shadow-[0_20px_45px_rgba(6,12,24,0.65)]"
                      : "border-white/10 bg-white/5 text-[#dfdecf] hover:border-[#1086ad]/40 hover:bg-white/10",
                  )}
                >
                  <span
                    className="text-sm font-semibold uppercase tracking-wide text-white"
                    style={{ fontFamily: "var(--font-condensed)" }}
                  >
                    {session.title}
                  </span>
                  <p className="line-clamp-2 text-xs text-[#9ba3c0]">
                    {lastMessage || "Sem mensagens ainda. Clique para abrir."}
                  </p>
                  <div className="text-[10px] uppercase tracking-[0.35em] text-[#5c6383]">
                    <span>{sessionMessages.length} mensagens · </span>
                    <span>{createdLabel}</span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </section>

      <div className="rounded-2xl border border-white/15 bg-white/5 p-4 text-sm text-[#9ba3c0]">
        <p className="text-[11px] uppercase tracking-[0.35em] text-[#7f8baf]">Pesquisa detalhada</p>
        <p className="mt-1 text-sm text-[#dfdecf]">Consulte todo o histórico para recuperar missões antigas.</p>
        <Link
          href="/history"
          className="mt-3 inline-flex items-center justify-center rounded-full border border-white/20 px-4 py-1 text-[11px] uppercase tracking-[0.35em] text-white transition hover:border-[#1086ad] hover:bg-[#1086ad]/10"
        >
          Abrir histórico
        </Link>
      </div>
    </aside>
  );
}
