import { API_BASE_URL } from "../config";

const buildApiUrl = (path) => `${API_BASE_URL}${path}`;

async function readErrorMessage(response) {
  try {
    const payload = await response.json();
    return payload.detail || payload.message || response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function predictFracture(file, { signal } = {}) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(buildApiUrl("/predict"), {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}
