"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

interface AuthContextValue {
  token: string | null;
  isReady: boolean;
  isLoggingIn: boolean;
  loginError: string | null;
  login: (passkey: string) => Promise<void>;
  logout: () => void;
}

const STORAGE_KEY = "genesis.auth.token";

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  useEffect(() => {
    const stored = typeof window !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
    if (stored) {
      setToken(stored);
    }
    setIsReady(true);
  }, []);

  const login = useCallback(async (passkey: string) => {
    setLoginError(null);
    setIsLoggingIn(true);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ passkey }),
      });
      if (!res.ok) {
        const message = res.status === 401 ? "Acesso não autorizado" : "Falha ao autenticar";
        await res.text(); // consome corpo para evitar leaks
        setLoginError(message);
        throw new Error(message);
      }
      const data = (await res.json()) as { token?: string };
      if (!data.token) {
        throw new Error("Resposta inválida do servidor");
      }
      localStorage.setItem(STORAGE_KEY, data.token);
      setToken(data.token);
      setLoginError(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Não foi possível autenticar";
      setLoginError(message);
      throw error;
    } finally {
      setIsLoggingIn(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setToken(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      isReady,
      isLoggingIn,
      loginError,
      login,
      logout,
    }),
    [token, isReady, isLoggingIn, loginError, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth precisa estar dentro de AuthProvider");
  }
  return context;
}
