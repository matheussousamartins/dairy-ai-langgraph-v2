"use client";

import clsx from "clsx";

interface AlertProps {
  variant?: "default" | "error" | "success";
  headline?: string;
  description?: string;
  children?: React.ReactNode;
  className?: string;
  onDismiss?: () => void;
}

export function Alert({ variant = "default", headline, description, children, className, onDismiss }: AlertProps) {
  return (
    <div
      className={clsx(
        "rounded-2xl border px-4 py-3 text-sm shadow-[0_15px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl",
        variant === "error" && "border-[#8b3dff]/45 bg-[#2b144f]/75 text-[#f5efff]",
        variant === "success" && "border-[#c0d044]/60 bg-[#1a220b]/80 text-[#e4f4a4]",
        variant === "default" && "border-white/12 bg-white/5 text-[#c9cdd6]",
        className,
      )}
    >
      <div className={clsx("flex items-start gap-3", onDismiss ? "justify-between" : "")}>
        <div className="flex-1">
          {headline ? <p className="text-[11px] font-semibold uppercase tracking-[0.35em] text-white/70">{headline}</p> : null}
          {description ? <p className="mt-1 text-sm">{description}</p> : null}
          {children}
        </div>
        {onDismiss ? (
          <button
            type="button"
            onClick={onDismiss}
            className="shrink-0 rounded-full p-0.5 opacity-50 transition hover:opacity-100"
            aria-label="Fechar"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden>
              <path d="M1 1l12 12M13 1L1 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
            </svg>
          </button>
        ) : null}
      </div>
    </div>
  );
}
