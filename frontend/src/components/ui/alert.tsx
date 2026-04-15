"use client";

import clsx from "clsx";

interface AlertProps {
  variant?: "default" | "error" | "success";
  headline?: string;
  description?: string;
  children?: React.ReactNode;
  className?: string;
}

export function Alert({ variant = "default", headline, description, children, className }: AlertProps) {
  return (
    <div
      className={clsx(
        "rounded-2xl border px-4 py-3 text-sm shadow-[0_15px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl",
        variant === "error" && "border-[#8267ad]/60 bg-[#221d56]/75 text-[#f5efff]",
        variant === "success" && "border-[#c0d044]/60 bg-[#1a220b]/80 text-[#e4f4a4]",
        variant === "default" && "border-white/12 bg-white/5 text-[#dfdecf]",
        className,
      )}
    >
      {headline ? <p className="text-[11px] font-semibold uppercase tracking-[0.35em] text-white/70">{headline}</p> : null}
      {description ? <p className="mt-1 text-sm">{description}</p> : null}
      {children}
    </div>
  );
}
