import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let cachedClient: SupabaseClient | null = null;

function getSupabaseServerEnv() {
  const url = process.env.SUPABASE_URL;
  const serviceRoleKey =
    process.env.SUPABASE_SERVICE_ROLE_KEY ??
    process.env.SUPABASE_SECRET_KEY;

  if (!url) {
    throw new Error("SUPABASE_URL não está configurado no frontend/.env.local.");
  }

  if (!serviceRoleKey) {
    throw new Error(
      "Configure SUPABASE_SERVICE_ROLE_KEY ou SUPABASE_SECRET_KEY no frontend/.env.local.",
    );
  }

  return { url, serviceRoleKey };
}

export function getSupabaseAdminClient() {
  if (cachedClient) {
    return cachedClient;
  }

  const { url, serviceRoleKey } = getSupabaseServerEnv();

  cachedClient = createClient(url, serviceRoleKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  });

  return cachedClient;
}
