export async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) { /* no JSON body */ }
    throw new Error(message);
  }
  return response.json();
}

export const getSources = () => api("/api/sources");
export const getSchema = sourceId => api(`/api/sources/${sourceId}/schema`);
export const getReview = payload => api("/api/review", { method: "POST", body: JSON.stringify(payload) });
export const getProjection = payload => api("/api/projection", { method: "POST", body: JSON.stringify(payload) });
export const getPoint = (sourceId, rowId, params) => {
  const query = new URLSearchParams(params);
  return api(`/api/points/${sourceId}/${rowId}?${query}`);
};
