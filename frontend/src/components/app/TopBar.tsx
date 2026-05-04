"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import { CircleHelp } from "lucide-react";
import { useGenesisCatalog, type AgentOption, type ModelOption } from "@/state/useGenesisUI";
import { useAuth } from "@/state/useAuth";
import { Button } from "@/components/ui/button";

interface TopBarProps {
  onToggleSidebar?: () => void;
}

const AGENT_STYLE: Record<string, { accent: string; dot: string }> = {
  orquestrador: { accent: "border-[#8b3dff]/70 bg-[#8b3dff]/10", dot: "bg-[#b06fff]" },
  "agente-1": { accent: "border-amber-500/40 bg-amber-500/8", dot: "bg-amber-400" },
  "agente-2": { accent: "border-purple-500/40 bg-purple-500/8", dot: "bg-purple-400" },
  "agente-3": { accent: "border-red-500/40 bg-red-500/8", dot: "bg-red-400" },
  "agente-4": { accent: "border-emerald-500/40 bg-emerald-500/8", dot: "bg-emerald-400" },
  "agente-5": { accent: "border-orange-500/40 bg-orange-500/8", dot: "bg-orange-400" },
  "agente-6": { accent: "border-teal-500/40 bg-teal-500/8", dot: "bg-teal-400" },
};

const MODEL_STYLE: Record<string, { accent: string; dot: string }> = {
  "gpt-4o-mini": { accent: "border-sky-500/40 bg-sky-500/8", dot: "bg-sky-400" },
  "gpt-4o": { accent: "border-cyan-500/40 bg-cyan-500/8", dot: "bg-cyan-400" },
  "gpt-4.1-mini": { accent: "border-indigo-500/40 bg-indigo-500/8", dot: "bg-indigo-400" },
  "gpt-4.1": { accent: "border-violet-500/40 bg-violet-500/8", dot: "bg-violet-400" },
  "gpt-4.1-nano": { accent: "border-fuchsia-500/40 bg-fuchsia-500/8", dot: "bg-fuchsia-400" },
  "claude-3.5-sonnet": { accent: "border-orange-500/40 bg-orange-500/8", dot: "bg-orange-400" },
  "llama-3.1-70b": { accent: "border-emerald-500/40 bg-emerald-500/8", dot: "bg-emerald-400" },
};

const DEFAULT_MODEL_STYLE = { accent: "border-white/15 bg-white/[0.04]", dot: "bg-[#b06fff]" };

function getModelFamily(model: ModelOption) {
  return model.family ?? model.provider ?? "Outros";
}

function getFamilyOrder(family: string) {
  if (family === "GPT-4o") return 0;
  if (family === "GPT-4.1") return 1;
  if (family === "Claude") return 2;
  if (family === "Llama") return 3;
  return 9;
}

function AgentCard({
  agent,
  active,
  fullWidth,
  onSelect,
}: {
  agent: AgentOption;
  active: boolean;
  fullWidth?: boolean;
  onSelect: () => void;
}) {
  const style = AGENT_STYLE[agent.id] ?? { accent: "border-white/10 bg-white/5", dot: "bg-white" };

  return (
    <button
      type="button"
      onClick={onSelect}
      className={clsx(
        "flex flex-col gap-2 rounded-2xl border px-4 py-5 text-left transition-all",
        fullWidth && "sm:col-span-2",
        active
          ? `${style.accent} text-white shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-white/10`
          : `${style.accent} text-[#c9cdd6] hover:brightness-125`,
      )}
    >
      <div className="flex items-center gap-2">
        <span className={clsx("h-2 w-2 rounded-full", style.dot)} />
        <span className="text-lg font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
          {agent.label}
        </span>
      </div>
      <span className="text-xs text-[#8f96ab]">{agent.id}</span>
      {fullWidth ? (
        <span className="text-xs text-[#8f96ab]">
          Deixe o sistema escolher automaticamente o agente mais adequado para sua pergunta.
        </span>
      ) : null}
      {active ? (
        <span className="mt-1 self-start rounded-full border border-white/30 px-3 py-0.5 text-[10px] uppercase tracking-[0.4em] text-white">
          Em uso
        </span>
      ) : null}
    </button>
  );
}

function ModelCard({
  model,
  active,
  isDefault,
  onSelect,
}: {
  model: ModelOption;
  active: boolean;
  isDefault: boolean;
  onSelect: () => void;
}) {
  const style = MODEL_STYLE[model.id] ?? { accent: "border-white/10 bg-white/5", dot: "bg-white" };
  const isSelectable = model.selectable !== false;
  const isReady = model.compatibilityStatus === "ready";

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!isSelectable}
      className={clsx(
        "flex flex-col gap-3 rounded-2xl border px-4 py-5 text-left transition-all",
        !isSelectable && "cursor-not-allowed opacity-65 saturate-50",
        active
          ? `${style.accent} text-white shadow-[0_20px_50px_rgba(0,0,0,0.5)] ring-1 ring-white/10`
          : `${style.accent} text-[#c9cdd6] hover:brightness-125`,
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className={clsx("h-2 w-2 rounded-full", style.dot)} />
          <span className="text-lg font-semibold uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
            {model.label}
          </span>
          <span
            title={model.setupHint ?? model.compatibilityMessage ?? "Sem instrucoes adicionais."}
            aria-label={model.setupHint ?? model.compatibilityMessage ?? "Sem instrucoes adicionais."}
            className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-white/15 bg-black/20 text-[#8f96ab] transition hover:border-[#8b3dff] hover:text-white"
          >
            <CircleHelp className="h-3.5 w-3.5" />
          </span>
        </div>
        <span
          className={clsx(
            "rounded-full border px-2.5 py-0.5 text-[10px] uppercase tracking-[0.3em]",
            isReady ? "border-emerald-500/30 text-emerald-300" : "border-amber-500/30 text-amber-300",
          )}
        >
          {isReady ? "Pronto" : "Requer setup"}
        </span>
      </div>
      <span className="text-xs uppercase tracking-[0.25em] text-[#8f96ab]">{model.provider ?? "Modelo"}</span>
      <span className="text-sm text-[#8f96ab]">{model.description ?? "Modelo disponivel para testes."}</span>
      <span className={clsx("text-xs", isReady ? "text-[#8baaa0]" : "text-amber-200/80")}>
        {model.compatibilityMessage ?? "Verifique a compatibilidade do backend."}
      </span>
      <div className="mt-1 min-h-[24px]">
        {active && isDefault ? (
          <span className="self-start rounded-full border border-cyan-400/40 bg-cyan-500/14 px-3 py-0.5 text-[10px] uppercase tracking-[0.35em] text-cyan-100 shadow-[0_0_0_1px_rgba(34,211,238,0.12)]">
            Selecionado e padrao
          </span>
        ) : isDefault ? (
          <span className="self-start rounded-full border border-violet-400/40 bg-violet-500/15 px-3 py-0.5 text-[10px] uppercase tracking-[0.35em] text-violet-200 shadow-[0_0_0_1px_rgba(167,139,250,0.12)]">
            Padrao do sistema
          </span>
        ) : active ? (
          <span className="self-start rounded-full border border-emerald-400/40 bg-emerald-500/12 px-3 py-0.5 text-[10px] uppercase tracking-[0.4em] text-emerald-200 shadow-[0_0_0_1px_rgba(52,211,153,0.1)]">
            Selecionado
          </span>
        ) : null}
      </div>
    </button>
  );
}

export function TopBar({ onToggleSidebar }: TopBarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const {
    agents,
    selectedAgentId,
    setSelectedAgentId,
    models,
    defaultModelId,
    selectedModelId,
    setSelectedModelId,
  } = useGenesisCatalog();
  const { logout, token } = useAuth();
  const [isMounted, setIsMounted] = useState(false);
  const [isAgentPanelOpen, setIsAgentPanelOpen] = useState(false);
  const [isModelPanelOpen, setIsModelPanelOpen] = useState(false);
  const hasAccess = Boolean(token);

  const handleLogout = useCallback(() => {
    logout();
    router.push("/");
  }, [logout, router]);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const selectedAgent = useMemo(
    () => agents.find((agent) => agent.id === selectedAgentId),
    [agents, selectedAgentId],
  );

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId],
  );
  const selectedModelStyle = selectedModel ? MODEL_STYLE[selectedModel.id] ?? DEFAULT_MODEL_STYLE : DEFAULT_MODEL_STYLE;
  const hasNonOpenAiModel = useMemo(
    () => models.some((model) => (model.provider ?? "").toLowerCase() !== "openai"),
    [models],
  );
  const groupedModels = useMemo(() => {
    const sorted = [...models].sort((left, right) => {
      const familyDiff = getFamilyOrder(getModelFamily(left)) - getFamilyOrder(getModelFamily(right));
      if (familyDiff !== 0) return familyDiff;
      if ((left.inputCost ?? 0) !== (right.inputCost ?? 0)) {
        return (left.inputCost ?? 0) - (right.inputCost ?? 0);
      }
      return left.label.localeCompare(right.label);
    });

    const groups = new Map<string, ModelOption[]>();
    for (const model of sorted) {
      const family = getModelFamily(model);
      const existing = groups.get(family) ?? [];
      existing.push(model);
      groups.set(family, existing);
    }
    return Array.from(groups.entries()).map(([family, items]) => ({ family, items }));
  }, [models]);

  const orchestrator = agents.find((agent) => agent.id === "orquestrador");
  const specialists = agents.filter((agent) => agent.id !== "orquestrador");

  return (
    <>
      <header className="sticky top-0 z-30 grid grid-cols-[1fr_auto_1fr] items-center gap-3 border-b border-[rgba(176,111,255,0.014)] bg-[linear-gradient(180deg,rgba(12,13,20,0.88),rgba(11,13,20,0.82))] px-4 py-3 text-[#c9cdd6] shadow-[0_1px_0_rgba(176,111,255,0.01),0_16px_34px_rgba(0,0,0,0.28)] backdrop-blur-xl sm:px-6 md:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="inline-flex h-11 w-11 items-center justify-center rounded-2xl border border-white/15 bg-white/[0.04] text-white transition hover:border-[#8b3dff]/40 hover:bg-[#8b3dff]/10 lg:hidden"
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
                  ? "border-white/16 bg-white/[0.035] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]"
                  : "border-white/8 bg-transparent text-[#8f96ab] hover:border-white/12 hover:bg-white/[0.025]",
              )}
            >
              Console
            </Link>
            <Link
              href="/tests"
              className={clsx(
                "rounded-full border px-3 py-2 transition sm:px-4",
                pathname === "/tests"
                  ? "border-white/16 bg-white/[0.035] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]"
                  : "border-white/8 bg-transparent text-[#8f96ab] hover:border-white/12 hover:bg-white/[0.025]",
              )}
            >
              Testes
            </Link>
            <button
              type="button"
              onClick={() => setIsModelPanelOpen(true)}
              className={clsx(
                "flex items-center gap-2 rounded-full border px-3 py-2 text-left transition sm:px-4",
                selectedModelStyle.accent,
                "text-[#c9cdd6] shadow-[inset_0_1px_0_rgba(255,255,255,0.015)] hover:brightness-105",
              )}
            >
              <span className="flex items-center gap-2 whitespace-nowrap">
                <span className={clsx("h-2 w-2 rounded-full", selectedModelStyle.dot)} />
                <span className="text-[10px] uppercase tracking-[0.28em] text-[#8f96ab]">Modelo IA</span>
                <span className="h-3.5 w-px bg-white/12" aria-hidden="true" />
                <span
                  className="text-sm font-semibold text-white"
                  style={{ fontFamily: "var(--font-condensed)" }}
                >
                  {selectedModel?.label ?? "Selecionar modelo"}
                </span>
              </span>
            </button>
          </nav>
        </div>

        <div className="flex flex-col items-center justify-center">
          <p className="text-[9px] uppercase tracking-[0.45em] text-[#8f96ab]">Cliente</p>
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
                className="inline-flex rounded-full border border-white/10 bg-white/[0.025] px-3 py-2 text-[10px] uppercase tracking-[0.35em] text-[#c9cdd6] transition hover:border-[#8b3dff]/28 hover:bg-[#8b3dff]/7 md:hidden"
              >
                Modelo
              </button>
              <button
                type="button"
                onClick={() => setIsAgentPanelOpen(true)}
                className="flex items-center gap-3 rounded-full border border-white/10 bg-white/[0.025] px-3 py-2 text-left transition shadow-[inset_0_1px_0_rgba(255,255,255,0.015)] hover:border-[#8b3dff]/28 hover:bg-[#8b3dff]/7"
              >
                <span className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-white/30 bg-transparent">
                  <Image src="/commandix-logo.png" alt="Selecionar agente" width={36} height={36} className="rounded-full" />
                </span>
                <div className="min-w-[160px]">
                  <p className="text-[10px] uppercase tracking-[0.4em] text-[#8f96ab]">Agente selecionado</p>
                  <p className="text-sm font-semibold text-white" style={{ fontFamily: "var(--font-condensed)" }}>
                    {selectedAgent?.label ?? selectedAgentId ?? "Selecionar"}
                  </p>
                </div>
              </button>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleLogout}
                className="border border-white/12 bg-white/[0.02] text-[10px] uppercase tracking-[0.4em] text-[#c9cdd6] hover:border-white/18 hover:bg-white/[0.04]"
              >
                Sair
              </Button>
            </>
          ) : (
            <div className="flex justify-start text-[11px] uppercase tracking-[0.35em] text-[#8f96ab] sm:justify-end">
              Aguardando autenticação
            </div>
          )}
        </div>
      </header>

      {isMounted && isAgentPanelOpen
        ? createPortal(
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(4,6,12,0.92)] px-6 py-10 backdrop-blur-2xl">
              <div className="relative w-full max-w-4xl rounded-[32px] border border-white/15 bg-[linear-gradient(180deg,rgba(23,26,34,0.96),rgba(11,13,20,0.98))] p-8 text-[#c9cdd6] shadow-[0_40px_90px_rgba(0,0,0,0.65)] ring-1 ring-white/10">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.4em] text-[#b06fff]/80">Seleção de Agente</p>
                    <h2 className="text-3xl font-black uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                      Escolha o Agente
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsAgentPanelOpen(false)}
                    className="rounded-full border border-white/20 px-4 py-2 text-sm uppercase tracking-[0.35em] text-[#c9cdd6] hover:border-[#8b3dff]"
                  >
                    Fechar
                  </button>
                </div>
                <div className="grid gap-4 sm:grid-cols-2">
                  {orchestrator ? (
                    <AgentCard
                      agent={orchestrator}
                      fullWidth
                      active={orchestrator.id === selectedAgentId}
                      onSelect={() => {
                        setSelectedAgentId(orchestrator.id);
                        setIsAgentPanelOpen(false);
                      }}
                    />
                  ) : null}
                  {specialists.map((agent) => (
                    <AgentCard
                      key={agent.id}
                      agent={agent}
                      active={agent.id === selectedAgentId}
                      onSelect={() => {
                        setSelectedAgentId(agent.id);
                        setIsAgentPanelOpen(false);
                      }}
                    />
                  ))}
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {isMounted && isModelPanelOpen
        ? createPortal(
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(4,6,12,0.92)] px-6 py-10 backdrop-blur-2xl">
              <div className="relative w-full max-w-4xl rounded-[32px] border border-white/15 bg-[linear-gradient(180deg,rgba(23,26,34,0.96),rgba(11,13,20,0.98))] p-8 text-[#c9cdd6] shadow-[0_40px_90px_rgba(0,0,0,0.65)] ring-1 ring-white/10">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <p className="text-[11px] uppercase tracking-[0.4em] text-[#b06fff]/80">Seleção de Modelo</p>
                    <h2 className="text-3xl font-black uppercase" style={{ fontFamily: "var(--font-condensed)" }}>
                      Escolha o Modelo
                    </h2>
                  </div>
                  <button
                    type="button"
                    onClick={() => setIsModelPanelOpen(false)}
                    className="rounded-full border border-white/20 px-4 py-2 text-sm uppercase tracking-[0.35em] text-[#c9cdd6] hover:border-[#8b3dff]"
                  >
                    Fechar
                  </button>
                </div>
                <div className="space-y-6">
                  {groupedModels.map((group) => (
                    <section key={group.family} className="space-y-3">
                      <div className="flex items-center gap-3">
                        <div>
                          <div className="flex items-center gap-3">
                            <h3
                              className="text-lg font-bold uppercase text-white"
                              style={{ fontFamily: "var(--font-condensed)" }}
                            >
                              {group.family}
                            </h3>
                            <span className="text-[10px] uppercase tracking-[0.35em] text-[#8f96ab]">
                              {group.items.length} modelos
                            </span>
                          </div>
                          <p className="mt-1 text-xs text-[#8f96ab]">
                            {group.items[0]?.familySubtitle ?? "Modelos disponíveis nesta categoria"}
                          </p>
                        </div>
                      </div>
                      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                        {group.items.map((model) => (
                          <ModelCard
                            key={model.id}
                            model={model}
                            active={model.id === selectedModelId}
                            isDefault={model.id === defaultModelId}
                            onSelect={() => {
                              if (model.selectable === false) return;
                              setSelectedModelId(model.id);
                              setIsModelPanelOpen(false);
                            }}
                          />
                        ))}
                      </div>
                    </section>
                  ))}
                </div>
                {hasNonOpenAiModel ? (
                  <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-[#8f96ab]">
                    Modelos OpenAI funcionam direto com `OPENAI_API_KEY`. Outros providers pedem gateway compatível no backend.
                  </div>
                ) : null}
                {selectedModel ? (
                  <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-[#8f96ab]">
                    Modelo ativo: <span className="text-white">{selectedModel.label}</span>
                    {selectedModel.compatibilityMessage ? (
                      <span className="block mt-1 text-xs text-[#8f96ab]">{selectedModel.compatibilityMessage}</span>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>,
            document.body,
          )
        : null}
    </>
  );
}


