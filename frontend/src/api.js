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

export async function createVector({ name, embedIds = [], customTexts = [] }) {
  const resp = await fetch("/vectors/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name,
      embed_ids: embedIds,
      custom_texts: customTexts
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

export async function listVectors({ ids }) {
  const resp = await fetch("/vectors/list", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids })
  });
  const data = await resp.json().catch(() => null);
  if (!resp.ok) {
    const message = data?.error || `Request failed (${resp.status})`;
    throw new Error(message);
  }
  return data;
}

export async function queryByVector({ vectorId, topK = 10, includeContext = true }) {
  const resp = await fetch("/query/by-vector", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      vector_id: vectorId,
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
