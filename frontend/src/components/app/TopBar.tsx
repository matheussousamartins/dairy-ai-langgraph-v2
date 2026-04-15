"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import { useGenesisUI } from "@/state/useGenesisUI";
import { useAuth } from "@/state/useAuth";
import { Button } from "@/components/ui/button";

interface TopBarProps {
  onToggleSidebar?: () => void;
}

export function TopBar({ onToggleSidebar }: TopBarProps) {
  const pathname = usePathname();
  const {
    models,
    selectedModelId,
    setSelectedModelId,
  } = useGenesisUI();
  const { logout, token } = useAuth();
  const [isMounted, setIsMounted] = useState(false);
  const [isModelPanelOpen, setIsModelPanelOpen] = useState(false);
  const hasAccess = Boolean(token);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId],
  );

  return (
    <>
      <header className="sticky top-0 z-30 grid grid-cols-[1fr_auto_1fr] items-center gap-3 bg-[rgba(9,14,26,0.9)] px-4 py-3 text-[#dfdecf] shadow-[0_1px_0_rgba(255,255,255,0.04),0_25px_80px_rgba(0,0,0,0.45)] backdrop-blur-2xl sm:px-6 md:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/20 bg-white/5 text-white transition hover:border-[#1086ad] hover:bg-[#1086ad]/15 lg:hidden"
            aria-label="Abrir menu"
          >
            <span className="flex flex-col gap-1.5">
              <span className="block h-0.5 w-5 bg-white" />
              <span className="block h-0.5 w-5 bg-white" />
              <span className="block h-0.5 w-5 bg-white" />
            </span>
          </button>
          <nav className="hidden items-center gap-2 text-[12px] uppercase tracking-[0.25em] md:flex">
            <Link
              href="/"
              className={clsx(
                "rounded-full border px-3 py-2 transition sm:px-4",
                pathname === "/"
                  ? "border-white/60 bg-white/10 text-white"
                  : "border-transparent text-[#9ba3c0] hover:border-white/20 hover:bg-white/5",
              )}
            >
              Console
            </Link>
            <Link
              href="/history"
              className={clsx(
                "rounded-full border px-3 py-2 transition sm:px-4",
                pathname === "/history"
                  ? "border-white/60 bg-white/10 text-white"
                  : "border-transparent text-[#9ba3c0] hover:border-white/20 hover:bg-white/5",
              )}
            >
              Histórico
            </Link>
          </nav>
        </div>

        {/* Centro — nome do cliente */}
        <div className="flex flex-col items-center justify-center">
          <p className="text-[9px] uppercase tracking-[0.45em] text-[#7f8baf]">Cliente</p>
          <p
            className="text-lg font-black uppercase tracking-widest text-white"
            style={{ fontFamily: "var(--font-condensed)" }}
          >
            DairyApp
          </p>
        </div>

        <div className="flex items-center justify-end gap-3 sm:gap-4">
          {hasAccess ? (
            <>
              <button
                type="button"
                onClick={() => setIsModelPanelOpen(true)}
                className="flex items-center gap-3 rounded-full border border-white/20 bg-white/5 px-3 py-2 text-left transition hover:border-[#1086ad] hover:bg-[#1086ad]/15"
              >
                <span className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-white/30 bg-transparent">
                  <Image src="/commandix-logo.png" alt="Selecionar agente" width={36} height={36} className="rounded-full" />
                </span>
                <div className="min-w-[160px]">
                  <p className="text-[10px] uppercase tracking-[0.4em] text-[#7f8baf]">Agente selecionado</p>
                  <p className="text-sm font-semibold text-white" style={{ fontFamily: "var(--font-condensed)" }}>
                    {selectedModel?.label ?? selectedModelId ?? "Selecionar"}
                  </p>
                </div>
              </button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={logout}
                className="border border-white/20 text-[10px] uppercase tracking-[0.4em]"
              >
                Sair
              </Button>
            </>
          ) : (
            <div className="flex justify-start text-[11px] uppercase tracking-[0.35em] text-[#7f8baf] sm:justify-end">
              Aguardando autenticação
            </div>
          )}
        </div>
      </header>
      {isMounted && isModelPanelOpen
        ? createPortal(
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(4,6,12,0.92)] px-6 py-10 backdrop-blur-2xl">
              <div className="relative w-full max-w-4xl rounded-[32px] border border-white/15 bg-gradient-to-br from-[#090f1c]/95 via-[#0e1525]/95 to-[#05080f]/95 p-8 text-[#dfdecf] shadow-[0_40px_90px_rgba(0,0,0,0.75)] ring-1 ring-white/10">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.4em] text-[#05adca]/70">Seleção de Agente</p>
                    <h2 className="text-3xl font-black uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                      Escolha o Agente
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsModelPanelOpen(false)}
                    className="rounded-full border border-white/20 px-4 py-2 text-sm uppercase tracking-[0.35em] text-[#dfdecf] hover:border-[#1086ad]"
                  >
                    Fechar
                  </button>
                </div>
                {(() => {
                  const AGENT_STYLE: Record<string, { accent: string; dot: string }> = {
                    "orquestrador": { accent: "border-[#05adca]/70 bg-[#05adca]/10", dot: "bg-[#05adca]" },
"agente-1":     { accent: "border-amber-500/40 bg-amber-500/8",   dot: "bg-amber-400" },
                    "agente-2":     { accent: "border-purple-500/40 bg-purple-500/8", dot: "bg-purple-400" },
                    "agente-3":     { accent: "border-red-500/40 bg-red-500/8",       dot: "bg-red-400" },
                    "agente-4":     { accent: "border-emerald-500/40 bg-emerald-500/8", dot: "bg-emerald-400" },
                    "agente-5":     { accent: "border-orange-500/40 bg-orange-500/8", dot: "bg-orange-400" },
                    "agente-6":     { accent: "border-teal-500/40 bg-teal-500/8",     dot: "bg-teal-400" },
                  };

                  const orchestrator = models.find((m) => m.id === "orquestrador");
                  const specialists  = models.filter((m) => m.id !== "orquestrador");

                  function AgentCard({ model, fullWidth }: { model: typeof models[0]; fullWidth?: boolean }) {
                    const active  = model.id === selectedModelId;
                    const style   = AGENT_STYLE[model.id] ?? { accent: "border-white/10 bg-white/5", dot: "bg-white" };
                    return (
                      <button
                        key={model.id}
                        onClick={() => { setSelectedModelId(model.id); setIsModelPanelOpen(false); }}
                        className={clsx(
                          "flex flex-col gap-2 rounded-2xl border px-4 py-5 text-left transition-all",
                          fullWidth && "sm:col-span-2",
                          active ? `${style.accent} text-white shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-white/10` : `${style.accent} text-[#dfdecf] hover:brightness-125`,
                        )}
                      >
                        <div className="flex items-center gap-2">
                          <span className={clsx("h-2 w-2 rounded-full", style.dot)} />
                          <span className="text-lg font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                            {model.label}
                          </span>
                        </div>
                        <span className="text-xs text-[#9ba3c0]">{model.id}</span>
                        {fullWidth && (
                          <span className="text-xs text-[#7f8baf]">Deixe o sistema escolher automaticamente o agente mais adequado para sua pergunta.</span>
                        )}
                        {active && (
                          <span className="mt-1 self-start rounded-full border border-white/30 px-3 py-0.5 text-[10px] uppercase tracking-[0.4em] text-white">
                            Em uso
                          </span>
                        )}
                      </button>
                    );
                  }

                  return (
                    <div className="grid gap-4 sm:grid-cols-2">
                      {orchestrator && <AgentCard model={orchestrator} fullWidth />}
                      {specialists.map((m) => <AgentCard key={m.id} model={m} />)}
                    </div>
                  );
                })()}
              </div>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}
