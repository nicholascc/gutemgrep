export async function postQuery({ query, topK = 10, includeContext = true }) {
  const resp = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      include_context: includeContext
    })
  });

  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const message = data?.error || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  if (data?.ok === false) {
    throw new Error(data?.error || "Request failed");
  }
  return data;
}

export async function getSavedQuery({ token }) {
  const resp = await fetch(`/api/query/${encodeURIComponent(token)}`, {
    method: "GET",
    headers: { Accept: "application/json" }
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const message = data?.error || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  if (data?.ok === false) {
    throw new Error(data?.error || "Request failed");
  }
  return data;
}

export async function getBookByEmbed({ embedId }) {
  const resp = await fetch(`/book/by-embed/${encodeURIComponent(embedId)}`, {
    method: "GET",
    headers: { Accept: "application/json" }
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const message = data?.error || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  if (data?.ok === false) {
    throw new Error(data?.error || "Request failed");
  }
  return data;
}
