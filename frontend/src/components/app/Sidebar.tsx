"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { memo, useCallback, useMemo } from "react";
import clsx from "clsx";
import { useGenesisConversation } from "@/state/useGenesisUI";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/state/useAuth";
import type { GenesisSession } from "@/state/useGenesisUI";

interface SidebarProps {
  isMobile?: boolean;
  onToggleCollapse?: () => void;
}

const ThreadCard = memo(function ThreadCard({
  session,
  active,
  preview,
  messageCount,
  createdLabel,
  onSelect,
}: {
  session: GenesisSession;
  active: boolean;
  preview: string;
  messageCount: number;
  createdLabel: string;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={clsx(
        "flex flex-col gap-2 rounded-[22px] border px-4 py-4 text-left transition-all shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]",
        active
          ? "border-[#8b3dff]/65 bg-[linear-gradient(180deg,rgba(139,61,255,0.12),rgba(43,20,79,0.16))] text-white shadow-[0_20px_45px_rgba(43,20,79,0.42)]"
          : "border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.025))] text-[#c9cdd6] hover:border-[#8b3dff]/30 hover:bg-white/[0.08]",
      )}
    >
      <span
        className="text-[0.95rem] font-semibold uppercase tracking-[0.02em] text-white"
        style={{ fontFamily: "var(--font-condensed)" }}
      >
        {session.title}
      </span>
      <p className="line-clamp-2 text-[12px] leading-5 text-[#8f96ab]">
        {preview || "Sem mensagens ainda. Clique para abrir."}
      </p>
      <div className="pt-1 text-[10px] uppercase tracking-[0.3em] text-[#676e83]">
        <span>{messageCount} mensagens · </span>
        <span>{createdLabel}</span>
      </div>
    </button>
  );
});

const SidebarHistoryCard = memo(function SidebarHistoryCard() {
  return (
    <div className="rounded-2xl border border-white/15 bg-[linear-gradient(180deg,rgba(255,255,255,0.05),rgba(255,255,255,0.025))] p-4 text-sm text-[#8f96ab] shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
      <p className="text-[11px] uppercase tracking-[0.35em] text-[#8f96ab]">Pesquisa detalhada</p>
      <p className="mt-1 text-sm text-[#c9cdd6]">Consulte todo o histórico para recuperar consultas antigas.</p>
      <Link
        href="/history"
        className="mt-3 inline-flex items-center justify-center rounded-full border border-white/18 bg-white/[0.03] px-4 py-1 text-[11px] uppercase tracking-[0.35em] text-white transition hover:border-[#8b3dff] hover:bg-[#8b3dff]/10"
      >
        Abrir histórico
      </Link>
    </div>
  );
});

export function Sidebar({ isMobile = false, onToggleCollapse }: SidebarProps) {
  const { isLoading, sessions, currentSessionId, selectSession, createSession, messagesBySession } = useGenesisConversation();
  const { token } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const hasAccess = Boolean(token);

  const recentSessions = useMemo(
    () => [...sessions].sort((a, b) => b.createdAt - a.createdAt),
    [sessions],
  );

  const navigateToConsoleIfNeeded = useCallback(() => {
    if (pathname !== "/") {
      router.push("/");
    }
  }, [pathname, router]);

  const handleCreateSession = useCallback(async () => {
    await createSession();
    navigateToConsoleIfNeeded();
  }, [createSession, navigateToConsoleIfNeeded]);

  const handleSelectSession = useCallback(async (sessionId: string) => {
    await selectSession(sessionId);
    navigateToConsoleIfNeeded();
  }, [navigateToConsoleIfNeeded, selectSession]);

  return (
    <aside
      className={clsx(
        "relative flex w-80 flex-col gap-6 border-white/[0.04] bg-[linear-gradient(180deg,rgba(18,18,27,0.96),rgba(12,13,20,0.99))] p-7 text-[#f5f6f8] shadow-[0_30px_80px_rgba(0,0,0,0.48)] backdrop-blur-2xl",
        isMobile
          ? "w-full max-w-[22rem] rounded-2xl border border-white/15 max-h-[calc(100vh-1.5rem)] overflow-y-auto"
          : "sticky top-0 h-screen border-r",
      )}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(176,111,255,0.06),transparent_28%)]" />
      <section className="space-y-3">
        <div>
          <div className="max-w-[16.75rem]">
            <p className="whitespace-nowrap text-[8.5px] uppercase tracking-[0.22em] text-[#b06fff]/80">Ambiente teste de agentes</p>
            <div className="mt-0.5 flex items-center justify-between gap-3">
              <h2 className="whitespace-nowrap text-[1.75rem] font-black uppercase leading-none text-white" style={{ fontFamily: "var(--font-condensed)" }}>
                Commandix AI
              </h2>
              {!isMobile && onToggleCollapse && (
                <button
                  type="button"
                  onClick={onToggleCollapse}
                  title="Recolher barra lateral"
                  className="inline-flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl border border-white/15 bg-white/5 text-[#8f96ab] transition-colors hover:border-[#8b3dff]/35 hover:text-[#b06fff]"
                >
                  <svg width="7" height="12" viewBox="0 0 7 12" fill="none" aria-hidden>
                    <path d="M6 1L1 6l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </div>
        <Button
          onClick={() => handleCreateSession().catch(console.error)}
          disabled={isLoading || !hasAccess}
          className="w-full justify-center border border-white/20 bg-white/5 text-sm uppercase tracking-[0.35em] text-white hover:border-[#8b3dff] hover:bg-[#8b3dff]/12 disabled:opacity-40"
        >
          Nova Sessão
        </Button>
      </section>

      <div className="h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent" />

      <section className="flex-1 overflow-hidden">
        <p className="mb-4 text-[11px] uppercase tracking-[0.28em] text-[#8f96ab]">Threads Recentes</p>
        <div className="-mr-3 flex h-full flex-col gap-2 overflow-y-auto pr-3 [scrollbar-color:rgba(176,111,255,0.26)_transparent] [scrollbar-width:thin] [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-[rgba(176,111,255,0.22)] [&::-webkit-scrollbar-thumb]:border [&::-webkit-scrollbar-thumb]:border-transparent [&::-webkit-scrollbar-thumb]:bg-clip-padding [&::-webkit-scrollbar-track]:bg-transparent hover:[&::-webkit-scrollbar-thumb]:bg-[rgba(176,111,255,0.34)]">
          {isLoading ? (
            <div className="space-y-2 rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-[#8f96ab]">
              Carregando threads...
            </div>
          ) : recentSessions.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/12 bg-white/5 p-6 text-center text-xs text-[#8f96ab]">
              Nenhuma thread ativa ainda.
            </div>
          ) : (
            recentSessions.map((session) => {
              const active = session.id === currentSessionId;
              const sessionMessages = messagesBySession[session.id] ?? [];
              const lastMessage = sessionMessages.slice(-1)[0]?.content ?? session.preview ?? "";
              const createdLabel = new Date(session.createdAt).toLocaleDateString("pt-BR");
              const messageCount = sessionMessages.length || session.messageCount || 0;
              return (
                <ThreadCard
                  key={session.id}
                  session={session}
                  active={active}
                  preview={lastMessage}
                  messageCount={messageCount}
                  createdLabel={createdLabel}
                  onSelect={() => handleSelectSession(session.id).catch(console.error)}
                />
              );
            })
          )}
        </div>
      </section>

      <SidebarHistoryCard />
    </aside>
  );
}


