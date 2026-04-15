import { NextRequest, NextResponse } from "next/server";
import { getThread } from "@/lib/thread-store";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest, context: { params: Promise<{ threadId: string }> }) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const { threadId } = await context.params;
  const thread = getThread(threadId);
  if (!thread) {
    return NextResponse.json({ error: "Thread not found" }, { status: 404 });
  }
  return NextResponse.json({ thread });
}
