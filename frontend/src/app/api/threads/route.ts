import { NextRequest, NextResponse } from "next/server";
import { createThread, listThreads } from "@/lib/thread-store";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const threads = listThreads();
  return NextResponse.json({ threads });
}

export async function POST(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }
  const thread = createThread();
  return NextResponse.json({ thread });
}
