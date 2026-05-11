function normalizeApiBaseUrl(value) {
  if (!value) {
    return "/api";
  }

  return value.replace(/\/$/, "");
}

export const API_BASE_URL = normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL);
