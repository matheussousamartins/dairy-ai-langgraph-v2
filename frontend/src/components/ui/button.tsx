import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost" | "outline";
  size?: "sm" | "md";
}

export function Button({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  const base =
    "inline-flex items-center justify-center rounded-xl font-semibold uppercase tracking-[0.25em] transition-all focus:outline-none focus:ring-2 focus:ring-[rgba(176,111,255,0.4)] focus:ring-offset-2 focus:ring-offset-[#0b0d14] disabled:cursor-not-allowed disabled:opacity-50";
  const variants = {
    primary:
      "bg-gradient-to-b from-[#b06fff] to-[#7b2cff] text-white shadow-[0_15px_35px_rgba(43,20,79,0.45)] hover:brightness-110 hover:shadow-[0_20px_45px_rgba(43,20,79,0.6)]",
    ghost: "bg-transparent text-[#c9cdd6] hover:text-white hover:bg-white/5",
    outline:
      "border border-white/20 text-[#c9cdd6] hover:border-[#8b3dff] hover:text-white hover:bg-[#8b3dff]/10",
  } as const;
  const sizes = {
    sm: "h-8 px-3 text-[11px]",
    md: "h-11 px-6 text-sm",
  } as const;

  return <button className={clsx(base, variants[variant], sizes[size], className)} {...props} />;
}
