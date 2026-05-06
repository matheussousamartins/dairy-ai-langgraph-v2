"use client";

import { memo, useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import { Minus, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { type GenesisMessage } from "@/state/useGenesisUI";
import { type TestErrorCategory, type TestVerdict } from "@/state/useThreadTesting";

interface EvaluationModalProps {
  message: GenesisMessage;
  initialVerdict: TestVerdict;
  initialComment?: string;
  initialErrorCategory?: TestErrorCategory;
  initialExpectedAnswer?: string;
  isSaving: boolean;
  onSave: (
    verdict: TestVerdict,
    payload: { comment: string; errorCategory?: TestErrorCategory; expectedAnswer?: string },
  ) => Promise<void>;
  onClose: () => void;
}

export const EvaluationModal = memo(function EvaluationModal({
  message,
  initialVerdict,
  initialComment,
  initialErrorCategory,
  initialExpectedAnswer,
  isSaving,
  onSave,
  onClose,
}: EvaluationModalProps) {
  const [selectedVerdict, setSelectedVerdict] = useState<TestVerdict>(initialVerdict);
  const [comment, setComment] = useState(initialComment ?? "");
  const [errorCategory, setErrorCategory] = useState<TestErrorCategory | "">(initialErrorCategory ?? "");
  const [expectedAnswer, setExpectedAnswer] = useState(initialExpectedAnswer ?? "");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleSave = useCallback(async () => {
    await onSave(selectedVerdict, {
      comment: comment.trim(),
      errorCategory: selectedVerdict === "correct" ? undefined : errorCategory || undefined,
      expectedAnswer: expectedAnswer.trim() || undefined,
    });
    setSaved(true);
    setTimeout(onClose, 700);
  }, [onSave, selectedVerdict, comment, errorCategory, expectedAnswer, onClose]);

  const preview =
    message.content.length > 140
      ? message.content.slice(0, 140) + "..."
      : message.content;

  const verdicts: { value: TestVerdict; label: string; icon: React.ReactNode; active: string; idle: string }[] = [
    {
      value: "correct",
      label: "Correta",
      icon: <ThumbsUp className="h-4 w-4" />,
      active: "border-emerald-400/55 bg-emerald-500/14 text-emerald-200 shadow-[0_0_24px_rgba(52,211,153,0.12)]",
      idle: "border-white/10 bg-white/[0.03] text-[#8f96ab] hover:border-emerald-400/30 hover:text-emerald-300",
    },
    {
      value: "partial",
      label: "Parcial",
      icon: <Minus className="h-4 w-4" />,
      active: "border-amber-400/55 bg-amber-500/14 text-amber-200 shadow-[0_0_24px_rgba(251,191,36,0.12)]",
      idle: "border-white/10 bg-white/[0.03] text-[#8f96ab] hover:border-amber-400/30 hover:text-amber-300",
    },
    {
      value: "incorrect",
      label: "Incorreta",
      icon: <ThumbsDown className="h-4 w-4" />,
      active: "border-rose-400/55 bg-rose-500/14 text-rose-200 shadow-[0_0_24px_rgba(251,113,133,0.12)]",
      idle: "border-white/10 bg-white/[0.03] text-[#8f96ab] hover:border-rose-400/30 hover:text-rose-300",
    },
  ];
  const errorCategories: Array<{ value: TestErrorCategory; label: string }> = [
    { value: "retrieval", label: "Busca RAG" },
    { value: "routing", label: "Roteamento" },
    { value: "consolidation", label: "Consolidação" },
    { value: "hallucination", label: "Alucinação" },
    { value: "missing_kb", label: "Falta na base" },
    { value: "regulatory_conflict", label: "Conflito regulatório" },
    { value: "wrong_scope", label: "Escopo errado" },
    { value: "ui", label: "Interface" },
    { value: "other", label: "Outro" },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/65 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-lg rounded-[28px] border border-[rgba(176,111,255,0.18)] bg-[linear-gradient(180deg,rgba(23,26,34,0.99),rgba(14,16,24,1))] p-6 shadow-[0_40px_120px_rgba(0,0,0,0.85)] ring-1 ring-[rgba(176,111,255,0.07)]">
        {/* Glow cap */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px rounded-t-[28px] bg-[linear-gradient(90deg,transparent,rgba(176,111,255,0.35),transparent)]" />
        <div className="pointer-events-none absolute inset-0 rounded-[28px] bg-[radial-gradient(circle_at_50%_0%,rgba(176,111,255,0.09),transparent_55%)]" />

        <div className="relative">
          {/* Header */}
          <div className="mb-5 flex items-start justify-between gap-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.42em] text-[#b06fff]/75">Avaliar resposta</p>
              <h3 className="mt-0.5 text-xl font-semibold text-white">Como foi esta resposta?</h3>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="shrink-0 rounded-full p-1.5 text-[#676e83] transition hover:bg-white/8 hover:text-white"
              aria-label="Fechar modal"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {/* Message preview */}
          <div className="mb-5 rounded-2xl border border-white/8 bg-white/[0.025] px-4 py-3">
            <p className="mb-1 text-[10px] uppercase tracking-[0.32em] text-[#676e83]">Resposta em avaliação</p>
            <p className="line-clamp-3 text-sm leading-relaxed text-[#9fa6bc]">{preview}</p>
          </div>

          {/* Verdict selection */}
          <div className="mb-5">
            <p className="mb-3 text-[10px] uppercase tracking-[0.38em] text-[#8f96ab]">Classificação</p>
            <div className="flex gap-2">
              {verdicts.map(({ value, label, icon, active, idle }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setSelectedVerdict(value)}
                  className={clsx(
                    "flex flex-1 items-center justify-center gap-2 rounded-2xl border py-3 text-[11px] uppercase tracking-[0.3em] transition",
                    selectedVerdict === value ? active : idle,
                  )}
                >
                  {icon}
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>

          {selectedVerdict !== "correct" ? (
            <div className="mb-5 grid gap-4 sm:grid-cols-2">
              <div>
                <p className="mb-2 text-[10px] uppercase tracking-[0.32em] text-[#8f96ab]">
                  Causa provável
                </p>
                <select
                  value={errorCategory}
                  onChange={(e) => setErrorCategory(e.target.value as TestErrorCategory | "")}
                  className="h-11 w-full rounded-2xl border border-white/8 bg-[rgba(255,255,255,0.025)] px-3 text-sm text-[#d8dcec] focus:border-[rgba(176,111,255,0.36)] focus:outline-none focus:ring-2 focus:ring-[rgba(139,61,255,0.12)]"
                >
                  <option value="">Sem classificar</option>
                  {errorCategories.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <p className="mb-2 text-[10px] uppercase tracking-[0.32em] text-[#8f96ab]">
                  Resposta esperada
                </p>
                <textarea
                  value={expectedAnswer}
                  onChange={(e) => setExpectedAnswer(e.target.value)}
                  placeholder="Opcional"
                  rows={2}
                  className="h-11 min-h-11 w-full resize-y rounded-2xl border border-white/8 bg-[rgba(255,255,255,0.025)] px-3 py-2 text-sm text-[#d8dcec] placeholder:text-[#414861] focus:border-[rgba(176,111,255,0.36)] focus:outline-none focus:ring-2 focus:ring-[rgba(139,61,255,0.12)]"
                />
              </div>
            </div>
          ) : null}

          {/* Comment */}
          <div className="mb-6">
            <p className="mb-2 text-[10px] uppercase tracking-[0.38em] text-[#8f96ab]">
              Observação{" "}
              <span className="normal-case tracking-normal text-[#676e83]">— opcional</span>
            </p>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Ex: faltou profundidade, boa resposta mas sem referência, agente escolheu foco errado..."
              rows={3}
              className="w-full resize-none rounded-2xl border border-white/8 bg-[rgba(255,255,255,0.025)] px-4 py-3 text-sm text-[#d8dcec] placeholder:text-[#414861] focus:border-[rgba(176,111,255,0.36)] focus:outline-none focus:ring-2 focus:ring-[rgba(139,61,255,0.12)]"
            />
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-[11px] uppercase tracking-[0.32em] text-[#676e83] transition hover:text-white"
            >
              Cancelar
            </button>
            <Button
              type="button"
              disabled={isSaving || saved}
              onClick={() => {
                void handleSave();
              }}
              className={clsx(
                "min-w-[140px] justify-center transition",
                saved && "border-emerald-600/50 bg-emerald-700/60 hover:bg-emerald-700/60",
              )}
            >
              {saved ? "Salvo ✓" : isSaving ? "Salvando..." : "Salvar avaliação"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
});
