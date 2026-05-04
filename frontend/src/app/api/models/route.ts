import { NextRequest, NextResponse } from "next/server";
import { resolveEnabledModelCatalog } from "@/lib/model-catalog";
import { backend, buildBackendHeaders } from "@/lib/dairy-backend";
import { ensureAuthorized, unauthorizedResponse } from "@/lib/server-auth";

interface BackendModelStatus {
  id: string;
  provider?: string;
  compatibility_status?: "ready" | "requires_adapter";
  compatibility_message?: string;
  setup_hint?: string;
  selectable?: boolean;
}

interface BackendModelsStatusResponse {
  models?: BackendModelStatus[];
  default_model?: string;
}

export async function GET(req: NextRequest) {
  if (!ensureAuthorized(req)) {
    return unauthorizedResponse();
  }

  const allowedIds = (process.env.ALLOWED_CHAT_MODELS ?? process.env.LLM_MODEL ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

  const fallbackModels = resolveEnabledModelCatalog(allowedIds);
  let effectiveModelIds = fallbackModels.map((model) => model.id);
  let backendStatusById = new Map<string, BackendModelStatus>();
  let defaultModelId = process.env.LLM_MODEL ?? allowedIds[0] ?? "";

  try {
    const response = await fetch(backend("/console/models/status"), {
      method: "GET",
      headers: buildBackendHeaders(),
      cache: "no-store",
    });

    if (response.ok) {
      const data = (await response.json()) as BackendModelsStatusResponse;
      const backendModels = data.models ?? [];
      backendStatusById = new Map(backendModels.map((item) => [item.id, item]));
      if (backendModels.length > 0) {
        effectiveModelIds = backendModels.map((item) => item.id);
      }
      if (data.default_model) {
        defaultModelId = data.default_model;
      }
    }
  } catch {
    // Fallback silencioso para o catalogo local se o backend estiver indisponivel.
  }

  const models = resolveEnabledModelCatalog(effectiveModelIds).map((model) => {
    const backendStatus = backendStatusById.get(model.id);
    return {
      id: model.id,
      label: model.label,
      provider: backendStatus?.provider ?? model.provider,
      description: model.description,
      family: model.family,
      family_subtitle: model.familySubtitle,
      compatibility_status:
        backendStatus?.compatibility_status ?? model.compatibilityStatus ?? "requires_adapter",
      compatibility_message:
        backendStatus?.compatibility_message ??
        model.compatibilityMessage ??
        "Não foi possível validar o backend em tempo real.",
      selectable: backendStatus?.selectable ?? model.selectable ?? false,
      setup_hint:
        backendStatus?.setup_hint ??
        model.setupHint ??
        "Não foi possível validar o backend em tempo real.",
      input_cost: model.input_cost ?? 0,
      output_cost: model.output_cost ?? 0,
    };
  });

  return NextResponse.json({ models, default_model: defaultModelId });
}
