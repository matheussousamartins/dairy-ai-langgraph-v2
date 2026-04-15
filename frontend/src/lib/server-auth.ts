import "server-only";
import { createHmac, timingSafeEqual } from "node:crypto";
import { NextRequest, NextResponse } from "next/server";

interface ConsoleTokenPayload {
  sub: "console";
  iat: number;
  exp: number;
  v: 1;
}

const DEFAULT_TTL_SECONDS = 60 * 60 * 8; // 8h

function base64UrlEncode(input: Buffer | string) {
  return Buffer.from(input)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

function base64UrlDecode(input: string) {
  const normalized = input.replace(/-/g, "+").replace(/_/g, "/");
  const padLength = (4 - (normalized.length % 4)) % 4;
  return Buffer.from(normalized + "=".repeat(padLength), "base64");
}

function getAuthSecret() {
  const secret = (process.env.CONSOLE_AUTH_SECRET ?? "").trim();
  if (!secret || secret.length < 32) {
    throw new Error("CONSOLE_AUTH_SECRET precisa ter no minimo 32 caracteres.");
  }
  return secret;
}

function getAllowedPasskeys() {
  const fromSingle = (process.env.CONSOLE_PASSKEY ?? "").trim();
  const fromList = (process.env.CONSOLE_PASSKEYS ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const merged = [...new Set([fromSingle, ...fromList].filter(Boolean))];
  return merged;
}

function sign(data: string, secret: string) {
  return createHmac("sha256", secret).update(data).digest();
}

function safeEqual(a: string, b: string) {
  const aBuf = Buffer.from(a);
  const bBuf = Buffer.from(b);
  if (aBuf.length !== bBuf.length) return false;
  return timingSafeEqual(aBuf, bBuf);
}

export function isAllowedPasskey(passkey: string) {
  const allowed = getAllowedPasskeys();
  if (!allowed.length) {
    return false;
  }
  return allowed.some((item) => safeEqual(item, passkey));
}

export function issueConsoleToken() {
  const secret = getAuthSecret();
  const now = Math.floor(Date.now() / 1000);
  const ttl = Number(process.env.CONSOLE_SESSION_TTL_SECONDS ?? DEFAULT_TTL_SECONDS);
  const payload: ConsoleTokenPayload = {
    sub: "console",
    iat: now,
    exp: now + Math.max(60, ttl),
    v: 1,
  };
  const header = { alg: "HS256", typ: "JWT" };
  const encodedHeader = base64UrlEncode(JSON.stringify(header));
  const encodedPayload = base64UrlEncode(JSON.stringify(payload));
  const signingInput = `${encodedHeader}.${encodedPayload}`;
  const signature = base64UrlEncode(sign(signingInput, secret));
  return `${signingInput}.${signature}`;
}

export function verifyConsoleToken(token: string): ConsoleTokenPayload | null {
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  const [encodedHeader, encodedPayload, encodedSignature] = parts;
  const secret = getAuthSecret();
  const signingInput = `${encodedHeader}.${encodedPayload}`;
  const expectedSignature = base64UrlEncode(sign(signingInput, secret));
  if (!safeEqual(expectedSignature, encodedSignature)) {
    return null;
  }

  try {
    const payload = JSON.parse(base64UrlDecode(encodedPayload).toString("utf-8")) as ConsoleTokenPayload;
    const now = Math.floor(Date.now() / 1000);
    if (payload.sub !== "console" || payload.v !== 1 || payload.exp <= now) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}

export function getBearerToken(req: NextRequest) {
  const raw = req.headers.get("authorization")?.trim() ?? "";
  const [scheme, value] = raw.split(/\s+/, 2);
  if (!scheme || !value || scheme.toLowerCase() !== "bearer") return null;
  return value;
}

export function ensureAuthorized(req: NextRequest) {
  const token = getBearerToken(req);
  if (!token) return false;
  try {
    return Boolean(verifyConsoleToken(token));
  } catch {
    return false;
  }
}

export function unauthorizedResponse() {
  return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
}
