import { clear, el, formatCurrency } from './dom.js';

export function buildTable({ columns, rows, editable = false }) {
  const table = el('table');
  const thead = el('thead');
  const headerRow = el('tr');
  columns.forEach((col) => {
    headerRow.append(el('th', { textContent: col }));
  });
  thead.append(headerRow);
  table.append(thead);

  const tbody = el('tbody');
  rows.forEach((row, rowIndex) => {
    const tr = el('tr');
    columns.forEach((col) => {
      const value = row[col];
      const cell = el('td', {
        textContent:
          value == null
            ? ''
            : typeof value === 'object'
            ? JSON.stringify(value)
            : value,
      });
      if (editable) {
        cell.contentEditable = col === 'Year' ? 'false' : 'true';
        cell.dataset.column = col;
        cell.dataset.rowIndex = rowIndex;
      }
      tr.append(cell);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  return table;
}

export function mountTable(wrapper, table) {
  clear(wrapper);
  wrapper.append(table);
}

export function tableToData(table, columns) {
  const rows = [];
  Array.from(table.querySelectorAll('tbody tr')).forEach((tr) => {
    const rowData = {};
    Array.from(tr.children).forEach((td, idx) => {
      const key = columns[idx];
      const raw = td.textContent.trim();
      rowData[key] = raw === '' ? null : coerceValue(raw);
    });
    rows.push(rowData);
  });
  return rows;
}

function coerceValue(value) {
  if (value === null || value === undefined) return value;
  const trimmed = String(value).trim();
  if (trimmed === '') return null;
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) || (trimmed.startsWith('[') && trimmed.endsWith(']')))
  {
    try {
      return JSON.parse(trimmed);
    } catch (err) {
      return trimmed;
    }
  }
  if (Number.isFinite(Number(trimmed))) {
    return Number(trimmed);
  }
  return trimmed;
}

export function renderMetricGrid(container, metrics) {
  clear(container);
  if (!metrics || metrics.length === 0) {
    container.append(el('p', { className: 'status', textContent: 'No metrics available.' }));
    return;
  }
  const grid = el('div', { className: 'metric-grid' });
  metrics.forEach((metric) => {
    const value = metric.value ?? metric.current ?? metric.amount ?? metric.metric_value ?? metric.metricValue;
    grid.append(
      el('div', {
        className: 'metric-card',
        children: [
          el('span', { className: 'metric-label', textContent: metric.metric || metric.label || metric.name || 'Metric' }),
          el('strong', { textContent: typeof value === 'number' ? formatCurrency(value) : String(value ?? '—') }),
          metric.description ? el('p', { className: 'status', textContent: metric.description }) : null,
        ],
      })
    );
  });
  container.append(grid);
}

export function renderKeyValueTable(container, data) {
  clear(container);
  if (!data || Object.keys(data).length === 0) {
    container.append(el('p', { className: 'status', textContent: 'No data available.' }));
    return;
  }
  const columns = ['Metric', 'Value'];
  const rows = Object.entries(data).map(([key, value]) => ({ Metric: key, Value: value }));
  const table = buildTable({ columns, rows });
  mountTable(container, table);
}

export function renderScheduleTable(container, schedule) {
  clear(container);
  if (!schedule || schedule.length === 0) {
    container.append(el('p', { className: 'status', textContent: 'No schedule available.' }));
    return;
  }
  const columns = Object.keys(schedule[0]);
  const table = buildTable({ columns, rows: schedule });
  mountTable(container, table);
}
