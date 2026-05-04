import { NextRequest, NextResponse } from "next/server";
import { deriveConsoleThreadOwnerId, getThreadWithTraces } from "@/lib/thread-store";
import { ensureAuthorized, getBearerToken, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const token = getBearerToken(req);
  if (!token) return unauthorizedResponse();
  const ownerId = deriveConsoleThreadOwnerId(token);
  const { threadId } = await context.params;
  const thread = await getThreadWithTraces(ownerId, threadId);
  if (!thread) {
    return NextResponse.json({ error: "Thread not found" }, { status: 404 });
  }
  return NextResponse.json({ thread });
}
