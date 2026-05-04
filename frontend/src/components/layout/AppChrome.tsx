"use client";

import { ReactNode, useState, useEffect } from "react";
import clsx from "clsx";
import { usePathname, useRouter } from "next/navigation";
import { Sidebar } from "@/components/app/Sidebar";
import { TopBar } from "@/components/app/TopBar";
import { AttributionBeacon } from "@/components/AttributionBeacon";
import { useAuth } from "@/state/useAuth";

interface AppChromeProps {
  children: ReactNode;
}

export function AppChrome({ children }: AppChromeProps) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const pathname = usePathname();
  const router = useRouter();
  const { token, isReady } = useAuth();

  useEffect(() => {
    const stored = localStorage.getItem("sidebar-collapsed");
    if (stored === "true") setIsSidebarCollapsed(true);
  }, []);

  const toggleDesktopSidebar = () => {
    setIsSidebarCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("sidebar-collapsed", String(next));
      return next;
    });
  };

  const closeSidebar = () => setIsSidebarOpen(false);
  const isTermsPage = pathname?.startsWith("/termos");
  const isHomePage = pathname === "/";

  useEffect(() => {
    if (!isReady) return;
    if (!token && !isHomePage && !isTermsPage) {
      router.replace("/");
    }
  }, [isReady, token, isHomePage, isTermsPage, router]);

  if (isTermsPage) {
    return (
      <div className="relative min-h-screen bg-[var(--cmdx-bg)] text-[var(--cmdx-text)]">
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
        {/* Desktop sidebar — collapsible */}
        <div
          className={clsx(
            "hidden lg:block flex-shrink-0 overflow-hidden transition-all duration-300 ease-in-out",
            isSidebarCollapsed ? "w-0" : "w-80",
          )}
        >
          <Sidebar onToggleCollapse={toggleDesktopSidebar} />
        </div>

        <div className="flex flex-1 flex-col lg:h-full">
          <TopBar onToggleSidebar={() => setIsSidebarOpen(true)} />
          <main className="flex-1 overflow-y-auto backdrop-blur-sm">{children}</main>
        </div>

        {/* Desktop expand button — fixed at left edge, only when collapsed */}
        <button
          type="button"
          onClick={toggleDesktopSidebar}
          title={isSidebarCollapsed ? "Expandir barra lateral" : "Recolher barra lateral"}
          className={clsx(
            "fixed left-3 top-1/2 z-40 hidden -translate-y-1/2 items-center justify-center rounded-xl border border-white/15 bg-[#171a22]/95 backdrop-blur-sm transition-all duration-300 hover:border-[#8b3dff]/40 hover:text-[#b06fff] lg:flex",
            isSidebarCollapsed
              ? "h-10 w-10 text-[#8f96ab]"
              : "pointer-events-none opacity-0 h-0 w-0 overflow-hidden",
          )}
        >
          <svg width="7" height="12" viewBox="0 0 7 12" fill="none" aria-hidden>
            <path d="M1 1l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>

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
            className="absolute right-3 top-3 rounded-full border border-white/20 bg-black/30 px-3 py-1 text-[11px] uppercase tracking-[0.35em] text-white hover:border-[#8b3dff]"
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
