const DEFAULT_BACKEND_BASE = "http://127.0.0.1:8000";

export const langgraphBaseUrl =
  process.env.LANGGRAPH_API_BASE ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  DEFAULT_BACKEND_BASE;
