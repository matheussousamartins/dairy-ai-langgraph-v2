"use client";

import { ReactNode, useState } from "react";
import clsx from "clsx";
import { usePathname } from "next/navigation";
import { Sidebar } from "@/components/app/Sidebar";
import { TopBar } from "@/components/app/TopBar";
import { AttributionBeacon } from "@/components/AttributionBeacon";
import { useAuth } from "@/state/useAuth";

interface AppChromeProps {
  children: ReactNode;
}

export function AppChrome({ children }: AppChromeProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const pathname = usePathname();
  const { token } = useAuth();

  const closeSidebar = () => setIsSidebarOpen(false);
  const isTermsPage = pathname?.startsWith("/termos");
  const isHomePage = pathname === "/";

  if (isTermsPage) {
    return (
      <div className="relative min-h-screen bg-gradient-to-br from-[#05080f] via-[#0a1426] to-[#080d18] text-[#dfdecf]">
        <div className="hawk-grid absolute inset-0" aria-hidden />
        <main className="relative mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">{children}</main>
      </div>
    );
  }

  if (isHomePage && !token) {
    return (
      <div className="relative h-screen overflow-hidden">
        <div className="hawk-grid absolute inset-0" aria-hidden />
        <AttributionBeacon />
        <main className="h-full overflow-y-auto backdrop-blur-sm">{children}</main>
      </div>
    );
  }

  return (
    <div className="relative h-screen overflow-hidden">
      <div className="hawk-grid absolute inset-0" aria-hidden />
      <AttributionBeacon />
      <div className="relative flex h-full flex-col lg:flex-row">
        <div className="hidden lg:block">
          <Sidebar />
        </div>

        <div className="flex flex-1 flex-col lg:h-full">
          <TopBar onToggleSidebar={() => setIsSidebarOpen(true)} />
          <main className="flex-1 overflow-y-auto backdrop-blur-sm">{children}</main>
        </div>

        {/* Mobile sidebar */}
        <div
          className={clsx(
            "fixed inset-y-0 left-0 z-50 w-[min(90vw,22rem)] max-w-full transform transition-transform duration-300 lg:hidden",
            isSidebarOpen ? "translate-x-0" : "-translate-x-full",
          )}
          aria-hidden={!isSidebarOpen}
        >
          <Sidebar isMobile />
          <button
            type="button"
            onClick={closeSidebar}
            className="absolute right-3 top-3 rounded-full border border-white/20 bg-black/30 px-3 py-1 text-[11px] uppercase tracking-[0.35em] text-white hover:border-[#1086ad]"
          >
            Fechar
          </button>
        </div>

        {isSidebarOpen ? (
          <button
            type="button"
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
            onClick={closeSidebar}
            aria-label="Fechar navegação"
          />
        ) : null}
      </div>
    </div>
  );
}
