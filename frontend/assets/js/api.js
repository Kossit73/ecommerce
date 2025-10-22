const apiState = {
  baseUrl: '',
};

export function setApiBase(url) {
  if (!url) {
    apiState.baseUrl = '';
    return;
  }
  try {
    const parsed = new URL(url, window.location.origin);
    apiState.baseUrl = parsed.origin + parsed.pathname.replace(/\/$/, '');
  } catch (err) {
    apiState.baseUrl = url.replace(/\/$/, '');
  }
}

export function getApiBase() {
  return apiState.baseUrl || '';
}

export async function apiRequest(path, { method = 'GET', body, headers } = {}) {
  const base = getApiBase();
  const target = `${base}${path.startsWith('/') ? path : `/${path}`}`;
  const init = {
    method,
    headers: {
      ...(body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...headers,
    },
    body: body instanceof FormData ? body : body != null ? JSON.stringify(body) : undefined,
    credentials: 'include',
  };

  const response = await fetch(target, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  if (contentType.includes('application/vnd.openxmlformats')) {
    return response.blob();
  }
  return response.text();
}

export async function apiGet(path) {
  return apiRequest(path, { method: 'GET' });
}

export async function apiPost(path, body) {
  return apiRequest(path, { method: 'POST', body });
}
