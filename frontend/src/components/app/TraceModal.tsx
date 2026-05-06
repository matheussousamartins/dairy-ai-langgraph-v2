"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { ChevronDown, ChevronUp, Clock3, Play, SearchCheck } from "lucide-react";
import { type TraceEvent } from "@/state/useGenesisUI";

const NODE_LABEL: Record<string, string> = {
  // V1 — Orchestrator
  prepare: "Preparar contexto",
  agent: "LLM decidindo",
  tools: "Executando ferramenta",
  classify: "Classificar domínio",
  route: "Planejar roteamento",
  execute: "Executar subagentes",
  respond_direct: "Resposta direta",
  consolidate: "Consolidar respostas",
  fallback_reclassify: "Reclassificação fallback",
  // V2 — Single-Agent
  analyze_query: "Analisar consulta",
  retrieve_context: "Buscar contexto",
  generate_answer: "Gerar resposta",
  validate_response: "Validar resposta",
};

const NODE_COLOR: Record<string, string> = {
  // V1
  classify: "text-sky-400",
  route: "text-sky-300",
  execute: "text-[#8b3dff]",
  consolidate: "text-violet-400",
  respond_direct: "text-emerald-400",
  fallback_reclassify: "text-amber-400",
  // V2
  analyze_query: "text-sky-400",
  retrieve_context: "text-amber-400",
  generate_answer: "text-[#8b3dff]",
  validate_response: "text-emerald-400",
};

function fmtMs(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

type RagChunk = { content: string; score?: number | null; source?: string };

function ChunkCard({ chunk, index }: { chunk: RagChunk; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = chunk.content.length > 320;

  return (
    <div className="rounded-xl border border-white/5 bg-[#080f1e] px-3 py-2.5 text-[11px]">
      <div className="mb-1.5 flex flex-wrap items-center gap-2 text-[10px]">
        <span className="font-semibold text-emerald-500/80">chunk {index + 1}</span>
        {chunk.score != null ? (
          <span
            className={clsx(
              "rounded px-1 py-0.5",
              chunk.score > 0.3 ? "text-emerald-400" : chunk.score > 0.08 ? "text-amber-400/80" : "text-[#5c6383]",
            )}
          >
            score: {chunk.score}
          </span>
        ) : null}
        {chunk.source ? (
          <span className="max-w-[260px] truncate text-[#4a5272]" title={chunk.source}>
            {chunk.source}
          </span>
        ) : null}
      </div>
      <p
        className={clsx(
          "whitespace-pre-wrap break-words leading-relaxed text-emerald-300/70",
          !expanded && isLong && "line-clamp-4",
        )}
      >
        {chunk.content}
      </p>
      {isLong ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="mt-1.5 inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-[#b06fff]/70 transition-colors hover:text-[#b06fff]"
        >
          {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          <span>{expanded ? "Recolher" : "Expandir"}</span>
        </button>
      ) : null}
    </div>
  );
}

export function TraceModal({ trace, onClose }: { trace: TraceEvent[]; onClose: () => void }) {
  const [copied, setCopied] = useState(false);

  const firstTs = trace[0]?.ts ?? Date.now();
  const lastTs = trace[trace.length - 1]?.ts ?? Date.now();
  const totalMs = lastTs - firstTs;
  const ragCount = trace.filter((event) => event.type === "tool_call" || event.type === "rag_result").length;
  const nodeCount = trace.filter((event) => event.type === "node_start").length;

  const ragPairs = useMemo(() => {
    const pairs = new Map<number, { ev: TraceEvent; latencyMs: number }>();
    const pending = new Map<string, number>();

    trace.forEach((event, index) => {
      if (event.type === "tool_call") {
        pending.set(event.tool ?? "", index);
      } else if (event.type === "tool_result") {
        const callIdx = pending.get(event.tool ?? "");
        if (callIdx !== undefined) {
          pairs.set(callIdx, { ev: event, latencyMs: event.ts - trace[callIdx].ts });
          pending.delete(event.tool ?? "");
        }
      }
    });

    return pairs;
  }, [trace]);

  const mergedResultIndices = useMemo(() => {
    const ids = new Set<number>();
    ragPairs.forEach(({ ev }) => {
      const idx = trace.findIndex((event) => event === ev);
      if (idx >= 0) ids.add(idx);
    });
    return ids;
  }, [trace, ragPairs]);

  const handleCopy = async () => {
    const lines: string[] = [
      "=== LOG DE EXECUÇÃO ===",
      `Latência total: ${fmtMs(totalMs)} · ${ragCount} buscas RAG · ${nodeCount} nós`,
      "",
    ];

    trace.forEach((event, index) => {
      if (mergedResultIndices.has(index)) return;

      const time = fmtTime(event.ts);
      const relative = index > 0 ? ` (+${fmtMs(event.ts - firstTs)})` : "";

      if (event.type === "node_start" || event.type === "node_end") {
        const label = NODE_LABEL[event.node ?? ""] ?? event.node ?? "";
        lines.push(`[${time}]${relative} ${event.type === "node_start" ? "INÍCIO" : "FIM"} — ${label.toUpperCase()}`);
      } else if (event.type === "rag_result") {
        lines.push(`[${time}]${relative} CHUNKS RAG — ${event.tool ?? ""}`);
        try {
          const chunks = JSON.parse(event.output ?? "") as RagChunk[];
          if (Array.isArray(chunks)) {
            chunks.forEach((chunk, chunkIndex) => {
              lines.push(
                `  chunk ${chunkIndex + 1}${chunk.score != null ? ` — score: ${chunk.score}` : ""}${chunk.source ? ` — ${chunk.source}` : ""}`,
              );
              lines.push(`    ${chunk.content}`);
            });
          }
        } catch {
          if (event.output) lines.push(`  ${event.output}`);
        }
      } else if (event.type === "tool_call") {
        lines.push(`[${time}]${relative} BUSCA RAG — ${event.tool ?? ""}`);
        if (event.input) lines.push(`  query: ${event.input}`);

        const pair = ragPairs.get(index);
        if (pair) {
          lines.push(`[${fmtTime(pair.ev.ts)}] RESULTADO (${fmtMs(pair.latencyMs)})`);
          try {
            const chunks = JSON.parse(pair.ev.output ?? "") as RagChunk[];
            if (Array.isArray(chunks)) {
              chunks.forEach((chunk, chunkIndex) => {
                lines.push(
                  `  chunk ${chunkIndex + 1}${chunk.score != null ? ` — score: ${chunk.score}` : ""}${chunk.source ? ` — ${chunk.source}` : ""}`,
                );
                lines.push(`    ${chunk.content}`);
              });
            }
          } catch {
            if (pair.ev.output) lines.push(`  ${pair.ev.output}`);
          }
        }
      }
    });

    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      // noop
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-end bg-black/40 p-4 backdrop-blur-sm sm:p-6" onClick={onClose}>
      <div
        className="relative flex h-[88vh] w-full max-w-2xl flex-col rounded-3xl border border-white/15 bg-[#05080f] shadow-[0_40px_100px_rgba(0,0,0,0.65)]"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between border-b border-white/10 px-5 py-4">
          <div>
            <p className="text-[10px] uppercase tracking-[0.4em] text-[#b06fff]/80">Processo interno</p>
            <h3 className="text-base font-bold uppercase text-white" style={{ fontFamily: "var(--font-condensed)" }}>
              Log da execução
            </h3>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[11px] text-[#5c6383]">
              <span className="inline-flex items-center gap-1 font-semibold text-[#9ba3c0]">
                <Clock3 className="h-3.5 w-3.5" />
                <span>{fmtMs(totalMs)}</span>
              </span>
              <span>·</span>
              <span>{ragCount} buscas RAG</span>
              <span>·</span>
              <span>{nodeCount} nós</span>
            </div>
          </div>
          <div className="ml-4 flex flex-shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              className={clsx(
                "rounded-full border px-3 py-1 text-[10px] uppercase tracking-[0.35em] transition-all",
                copied
                  ? "border-emerald-500/50 text-emerald-400"
                  : "border-white/15 text-[#8f96ab] hover:border-[#8b3dff]/40 hover:text-[#b06fff]",
              )}
            >
              {copied ? "Copiado" : "Copiar"}
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-full border border-white/15 px-3 py-1 text-[10px] uppercase tracking-[0.35em] text-[#9ba3c0] hover:border-white/30 hover:text-white"
            >
              Fechar
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4 font-mono text-xs">
          {trace.length === 0 ? (
            <p className="text-[#5c6383]">Nenhum evento registrado.</p>
          ) : (
            trace.map((event, index) => {
              if (mergedResultIndices.has(index)) return null;

              const time = fmtTime(event.ts);
              const relative = index > 0 ? `+${fmtMs(event.ts - firstTs)}` : "0ms";

              if (event.type === "node_start" || event.type === "node_end") {
                const isStart = event.type === "node_start";
                const label = NODE_LABEL[event.node ?? ""] ?? event.node ?? "";
                const color = isStart ? (NODE_COLOR[event.node ?? ""] ?? "text-[#8b3dff]") : "text-[#676e83]";

                return (
                  <div key={index} className="mb-2">
                    <div className={clsx("flex items-center gap-2", color)}>
                      <Play className={clsx("h-3.5 w-3.5", !isStart && "rotate-180")} />
                      <span className="text-[11px] uppercase tracking-wider">
                        {isStart ? "Início" : "Fim"} — {label}
                      </span>
                      <span className="ml-auto flex items-center gap-2.5 text-[#2d3348]">
                        <span className="text-[10px]">{relative}</span>
                        <span>{time}</span>
                      </span>
                    </div>
                  </div>
                );
              }

              if (event.type === "tool_call") {
                const pair = ragPairs.get(index);
                let chunks: RagChunk[] | null = null;
                if (pair?.ev.output) {
                  try {
                    const parsed = JSON.parse(pair.ev.output);
                    if (Array.isArray(parsed)) chunks = parsed;
                  } catch {
                    // raw output
                  }
                }

                return (
                  <div key={index} className="mb-5">
                    <div className="flex items-center gap-2 text-amber-400">
                      <SearchCheck className="h-3.5 w-3.5" />
                      <span className="text-[11px] uppercase tracking-wider">
                        Busca RAG — <span className="text-amber-300">{event.tool}</span>
                      </span>
                      <span className="ml-auto flex items-center gap-2.5 text-[#2d3348]">
                        {pair ? (
                          <span
                            className={clsx(
                              "rounded-full px-1.5 text-[10px]",
                              pair.latencyMs > 8000
                                ? "text-amber-500"
                                : pair.latencyMs > 3000
                                  ? "text-amber-400/60"
                                  : "text-emerald-500/60",
                            )}
                          >
                            {fmtMs(pair.latencyMs)}
                          </span>
                        ) : null}
                        <span>{time}</span>
                      </span>
                    </div>

                    {event.input ? (
                      <div className="ml-4 mt-1 rounded-xl border border-white/5 bg-[#0d1422] px-3 py-2 text-[11px] text-amber-300/80">
                        <span className="text-[#5c6383]">query › </span>
                        {event.input}
                      </div>
                    ) : null}

                    {chunks ? (
                      <div className="ml-4 mt-1.5 space-y-1.5">
                        {chunks.map((chunk, chunkIndex) => (
                          <ChunkCard key={chunkIndex} chunk={chunk} index={chunkIndex} />
                        ))}
                      </div>
                    ) : pair?.ev.output ? (
                      <pre className="ml-4 mt-1 whitespace-pre-wrap break-words rounded-xl border border-white/5 bg-[#080f1e] px-3 py-2 text-[11px] text-emerald-300/70">
                        {pair.ev.output}
                      </pre>
                    ) : null}
                  </div>
                );
              }

              if (event.type === "rag_result") {
                let chunks: RagChunk[] | null = null;
                if (event.output) {
                  try {
                    const parsed = JSON.parse(event.output);
                    if (Array.isArray(parsed)) chunks = parsed;
                  } catch {
                    // raw
                  }
                }
                return (
                  <div key={index} className="mb-5">
                    <div className="flex items-center gap-2 text-amber-400">
                      <SearchCheck className="h-3.5 w-3.5" />
                      <span className="text-[11px] uppercase tracking-wider">
                        Chunks RAG — <span className="text-amber-300">{event.tool}</span>
                      </span>
                      <span className="ml-auto text-[10px] text-[#2d3348]">{time}</span>
                    </div>
                    {chunks && chunks.length > 0 ? (
                      <div className="ml-4 mt-1.5 space-y-1.5">
                        {chunks.map((chunk, chunkIndex) => (
                          <ChunkCard key={chunkIndex} chunk={chunk} index={chunkIndex} />
                        ))}
                      </div>
                    ) : (
                      <div className="ml-4 mt-1 text-[11px] text-[#5c6383]">Nenhum chunk encontrado.</div>
                    )}
                  </div>
                );
              }

              return null;
            })
          )}
        </div>

        <div className="flex items-center justify-between border-t border-white/10 px-5 py-3 text-[10px] uppercase tracking-[0.35em] text-[#3a3f55]">
          <span>{trace.length} eventos · {ragCount} buscas RAG</span>
          <span className="inline-flex items-center gap-1">
            <Clock3 className="h-3.5 w-3.5" />
            <span>{fmtMs(totalMs)}</span>
          </span>
        </div>
      </div>
    </div>
  );
}
