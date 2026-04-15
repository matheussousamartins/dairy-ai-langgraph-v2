import { getAgentById } from "@/lib/agent-catalog";
import { langgraphBaseUrl } from "@/lib/config";

export interface DairyBackendResponse {
  response: string;
  agent_id?: number;
  agent_name?: string;
}

export function backend(path: string) {
  return `${langgraphBaseUrl}${path}`;
}

export function resolveAgentEndpoint(agentId: string) {
  const catalog = getAgentById(agentId);
  return catalog?.endpoint ?? "/webhook/orquestrador";
}

const backendApiKey = (process.env.BACKEND_API_KEY ?? "").trim();
const backendApiKeyHeader = (process.env.BACKEND_API_KEY_HEADER ?? "X-API-Key").trim();

export function buildBackendHeaders(authHeader?: string | null): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (authHeader) {
    headers.Authorization = authHeader;
  }
  if (backendApiKey) {
    headers[backendApiKeyHeader] = backendApiKey;
  }
  return headers;
}

export async function callDairyWebhook(
  agentId: string,
  message: string,
  sessionId: string,
  authHeader?: string | null,
): Promise<DairyBackendResponse> {
  const endpoint = resolveAgentEndpoint(agentId);
  const headers = buildBackendHeaders(authHeader);

  const resp = await fetch(backend(endpoint), {
    method: "POST",
    headers,
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
    cache: "no-store",
  });

  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(detail || `Backend retornou HTTP ${resp.status}`);
  }

  return (await resp.json()) as DairyBackendResponse;
}
