export function qs(selector, scope = document) {
  return scope.querySelector(selector);
}

export function qsa(selector, scope = document) {
  return Array.from(scope.querySelectorAll(selector));
}

export function clear(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

export function el(tag, options = {}) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.textContent != null) node.textContent = options.textContent;
  if (options.html != null) node.innerHTML = options.html;
  if (options.attrs) {
    Object.entries(options.attrs).forEach(([key, value]) => {
      if (value != null) {
        node.setAttribute(key, value);
      }
    });
  }
  if (options.children) {
    options.children.forEach((child) => {
      if (child != null) node.append(child);
    });
  }
  return node;
}

export function toast(message, { duration = 4000, tone = 'info' } = {}) {
  const container = qs('#toast');
  if (!container) return;
  container.textContent = message;
  container.dataset.tone = tone;
  container.classList.add('show');
  clearTimeout(container._timeout);
  container._timeout = setTimeout(() => {
    container.classList.remove('show');
  }, duration);
}

export function formatCurrency(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(Number(value));
}

export function formatPercent(value) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}
