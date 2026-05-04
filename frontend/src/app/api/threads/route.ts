import { NextRequest, NextResponse } from "next/server";
import { createThread, deriveConsoleThreadOwnerId, listThreads } from "@/lib/thread-store";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const token = getBearerToken(req);
  if (!token) return unauthorizedResponse();
  const ownerId = deriveConsoleThreadOwnerId(token);
  const threads = await listThreads(ownerId);
  return NextResponse.json({ threads });
}

export async function POST(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const token = getBearerToken(req);
  if (!token) return unauthorizedResponse();
  const ownerId = deriveConsoleThreadOwnerId(token);
  const thread = await createThread(ownerId);
  return NextResponse.json({ thread });
}
