import { NextRequest, NextResponse } from "next/server";
import { isAllowedPasskey, issueConsoleToken } from "@/lib/server-auth";

export async function POST(req: NextRequest) {
  const body = (await req.json()) as { passkey?: string };
  const passkey = (body?.passkey ?? "").trim();

  if (!passkey || !isAllowedPasskey(passkey)) {
    return NextResponse.json({ error: "Acesso nao autorizado" }, { status: 401 });
  }

  try {
    const token = issueConsoleToken();
    return NextResponse.json({ token });
  } catch {
    return NextResponse.json({ error: "Auth misconfigured" }, { status: 500 });
  }
}
