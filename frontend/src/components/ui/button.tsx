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
    "inline-flex items-center justify-center rounded-xl font-semibold uppercase tracking-[0.25em] transition-all focus:outline-none focus:ring-2 focus:ring-[rgba(16,134,173,0.4)] focus:ring-offset-2 focus:ring-offset-[#080e18] disabled:cursor-not-allowed disabled:opacity-50";
  const variants = {
    primary:
      "bg-gradient-to-r from-[#20336d] via-[#1a447c] to-[#1086ad] text-white shadow-[0_15px_35px_rgba(7,14,24,0.45)] hover:shadow-[0_20px_45px_rgba(8,20,40,0.6)]",
    ghost: "bg-transparent text-[#dfdecf] hover:text-white hover:bg-white/5",
    outline:
      "border border-white/20 text-[#dfdecf] hover:border-[#1086ad] hover:text-white hover:bg-[#1086ad]/10",
  } as const;
  const sizes = {
    sm: "h-8 px-3 text-[11px]",
    md: "h-11 px-6 text-sm",
  } as const;

  return <button className={clsx(base, variants[variant], sizes[size], className)} {...props} />;
}
