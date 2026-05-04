import clsx from "clsx";
import type { ButtonHTMLAttributes } from "react";

interface SwitchProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  checked: boolean;
  label?: string;
}

export function Switch({ checked, label, className, ...props }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className={clsx("flex items-center gap-3", className)}
      {...props}
    >
      <span
        className={clsx(
          "relative inline-flex h-7 w-14 flex-shrink-0 items-center rounded-full border transition-all",
          checked
            ? "border-[#8b3dff]/75 bg-gradient-to-r from-[#2b144f]/90 via-[#6d28f0]/88 to-[#b06fff]/85 shadow-[0_8px_20px_rgba(139,61,255,0.28)]"
            : "border-white/15 bg-white/5",
        )}
      >
        <span
          className={clsx(
            "inline-block h-5 w-5 transform rounded-full bg-[#f5f6f8] transition-transform shadow",
            checked ? "translate-x-7" : "translate-x-1",
          )}
        />
      </span>
      {label ? <span className="text-sm text-[#c9cdd6]">{label}</span> : null}
    </button>
  );
}
