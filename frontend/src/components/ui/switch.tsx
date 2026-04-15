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
            ? "border-[#1086ad]/80 bg-gradient-to-r from-[#20336d]/85 via-[#1a447c]/85 to-[#05adca]/80 shadow-[0_8px_20px_rgba(5,173,202,0.25)]"
            : "border-white/15 bg-white/5",
        )}
      >
        <span
          className={clsx(
            "inline-block h-5 w-5 transform rounded-full bg-[#dfdecf] transition-transform shadow",
            checked ? "translate-x-7" : "translate-x-1",
          )}
        />
      </span>
      {label ? <span className="text-sm text-[#dfdecf]">{label}</span> : null}
    </button>
  );
}
