import { setApiBase, getApiBase, apiGet, apiPost, apiRequest } from './api.js';
import { qs, qsa, clear, el, toast, formatCurrency } from './dom.js';
import {
  buildTable,
  mountTable,
  tableToData,
  renderMetricGrid,
  renderKeyValueTable,
  renderScheduleTable,
  coerceValue,
} from './tables.js';
import { renderPlotlyChart } from './charts.js';

const YEAR_COLUMN = 'Year';
const DEFAULT_SCENARIO_TYPE = 'Base Case';
const SCENARIO_DEFAULTS = {
  'Base Case': {
    conversion_rate_mult: 1.0,
    aov_mult: 1.0,
    cogs_mult: 1.0,
    interest_mult: 1.0,
    labor_mult: 1.0,
    material_mult: 1.0,
    markdown_mult: 1.0,
    political_risk: 0,
    env_impact: 0,
  },
  'Best Case': {
    conversion_rate_mult: 1.2,
    aov_mult: 1.1,
    cogs_mult: 0.95,
    interest_mult: 0.9,
    labor_mult: 0.9,
    material_mult: 0.9,
    markdown_mult: 0.9,
    political_risk: 2,
    env_impact: 2,
  },
  'Worst Case': {
    conversion_rate_mult: 0.8,
    aov_mult: 0.9,
    cogs_mult: 1.05,
    interest_mult: 1.2,
    labor_mult: 1.2,
    material_mult: 1.2,
    markdown_mult: 1.1,
    political_risk: 4,
    env_impact: 4,
  },
};

const SCENARIO_PARAM_LABELS = {
  conversion_rate_mult: 'Conversion rate multiplier',
  aov_mult: 'Average order value multiplier',
  cogs_mult: 'COGS multiplier',
  interest_mult: 'Interest multiplier',
  labor_mult: 'Labor multiplier',
  material_mult: 'Material multiplier',
  markdown_mult: 'Markdown multiplier',
  political_risk: 'Political risk',
  env_impact: 'Environmental impact',
};

function getScenarioDefaults(type = DEFAULT_SCENARIO_TYPE) {
  const defaults = SCENARIO_DEFAULTS[type] || SCENARIO_DEFAULTS[DEFAULT_SCENARIO_TYPE];
  return { ...defaults };
}

const ASSUMPTION_GROUPS = [
  {
    id: 'demand-pricing',
    title: 'Demand & pricing',
    description: 'Traffic, conversion, and pricing drivers used to project orders and revenue.',
    columns: [
      YEAR_COLUMN,
      'Email Traffic',
      'Organic Search Traffic',
      'Paid Search Traffic',
      'Affiliates Traffic',
      'Email Conversion Rate',
      'Organic Search Conversion Rate',
      'Paid Search Conversion Rate',
      'Affiliates Conversion Rate',
      'Average Item Value',
      'Number of Items per Order',
      'Average Markdown',
      'Average Promotion/Discount',
      'Churn Rate',
    ],
  },
  {
    id: 'unit-economics',
    title: 'Unit economics & acquisition costs',
    description: 'COGS, fulfillment, and acquisition cost assumptions powering gross margin.',
    columns: [
      YEAR_COLUMN,
      'COGS Percentage',
      'Freight/Shipping per Order',
      'Labor/Handling per Order',
      'Email Cost per Click',
      'Organic Search Cost per Click',
      'Paid Search Cost per Click',
      'Affiliates Cost per Click',
      'Other',
    ],
  },
  {
    id: 'overhead',
    title: 'Overhead & facilities',
    description: 'Rent schedules, professional fees, and location-specific warehouse inputs.',
    columns: [
      YEAR_COLUMN,
      'General Warehouse Rent',
      'Office Rent',
      'Rent Categories',
      'Warehouse2 Square Meters',
      'Warehouse2 Cost per SQM',
      'Warehouse2',
      'sun warehouse Square Meters',
      'sun warehouse Cost per SQM',
      'sun warehouse',
      'new warehouse Square Meters',
      'new warehouse Cost per SQM',
      'new warehouse',
      'Professional Fees',
      'Professional Fee Types',
      'Legal Cost',
      'Legal',
    ],
  },
  {
    id: 'staffing',
    title: 'Staffing structure',
    description: 'Direct, indirect, and part-time staffing capacity with loaded hourly rates.',
    columns: [
      YEAR_COLUMN,
      'Direct Staff Hours per Year',
      'Direct Staff Number',
      'Direct Staff Hourly Rate',
      'Direct Staff Total Cost',
      'Indirect Staff Hours per Year',
      'Indirect Staff Number',
      'Indirect Staff Hourly Rate',
      'Indirect Staff Total Cost',
      'Part-Time Staff Hours per Year',
      'Part-Time Staff Number',
      'Part-Time Staff Hourly Rate',
      'Part-Time Staff Total Cost',
    ],
  },
  {
    id: 'leadership-benefits',
    title: 'Leadership & benefits',
    description: 'Executive compensation and benefit assumptions rolled into operating expenses.',
    columns: [
      YEAR_COLUMN,
      'CEO Salary',
      'COO Salary',
      'CFO Salary',
      'Director of HR Salary',
      'CIO Salary',
      'Salaries, Wages & Benefits',
      'Pension Cost per Staff',
      'Pension Total Cost',
      'Medical Insurance Cost per Staff',
      'Medical Insurance Total Cost',
      'Child Benefit Cost per Staff',
      'Child Benefit Total Cost',
      'Car Benefit Cost per Staff',
      'Car Benefit Total Cost',
      'Total Benefits',
    ],
  },
  {
    id: 'working-capital',
    title: 'Working capital & depreciation',
    description: 'Depreciation schedules and cash conversion cycle levers.',
    columns: [
      YEAR_COLUMN,
      'Depreciation',
      'Accounts Receivable Days',
      'Inventory Days',
      'Accounts Payable Days',
      'Technology Development',
      'Office Equipment',
      'Technology Depreciation Years',
      'Office Equipment Depreciation Years',
    ],
  },
  {
    id: 'financing',
    title: 'Financing & capital actions',
    description: 'Interest, tax, and capital structure updates for each forecast year.',
    columns: [
      YEAR_COLUMN,
      'Interest',
      'Tax Rate',
      'Interest Rate',
      'Equity Raised',
      'Dividends Paid',
      'Debt Issued',
      'legal_2024 Cost',
      'legal_2024',
      'legal_2025 Cost',
      'legal_2025',
    ],
  },
  {
    id: 'assets',
    title: 'Capital assets',
    description: 'Long-lived asset balances and depreciation rates.',
    columns: [
      YEAR_COLUMN,
      'Asset_1_Name',
      'Asset_1_Amount',
      'Asset_1_Rate',
      'Asset_1_Depreciation',
      'Asset_1_NBV',
    ],
  },
  {
    id: 'debt',
    title: 'Debt schedules',
    description: 'Outstanding debt instruments, rates, and durations feeding amortization.',
    columns: [
      YEAR_COLUMN,
      'Debt_1_Name',
      'Debt_1_Amount',
      'Debt_1_Interest_Rate',
      'Debt_1_Duration',
    ],
  },
];

const state = {
  assumptions: [],
  originalAssumptions: [],
  assumptionColumns: getDefaultAssumptionColumns(),
  scenario: {
    type: DEFAULT_SCENARIO_TYPE,
    params: getScenarioDefaults(),
  },
  scenarioResult: null,
};

function init() {
  initializeApiBase();
  wireNavigation();
  wireInputPage();
  wireMetricsPage();
  wirePerformancePage();
  wireCashflowPage();
  wireSensitivityPage();
  wireAdvancedPage();
  renderScenarioParameters();
  renderScenarioResult(state.scenarioResult);
  loadScenarioDefaults(state.scenario.type, { silent: true }).catch(() => {});
  refreshWorkbook('Load Existing').catch(() => {});
  loadKeyMetrics().catch(() => {});
  loadFinancialSchedules().catch(() => {});
}

function wireNavigation() {
  qsa('.nav-link').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      qsa('.nav-link').forEach((b) => b.classList.toggle('active', b === btn));
      qsa('.panel').forEach((panel) => panel.classList.toggle('active', panel.id === target));
      if (target === 'metrics-section') {
        loadKeyMetrics().catch((err) => toast(err.message, { tone: 'error' }));
      }
      if (target === 'performance-section') {
        loadFinancialSchedules().catch((err) => toast(err.message, { tone: 'error' }));
        loadPerformanceCharts().catch((err) => toast(`Performance charts failed: ${err.message}`, { tone: 'error' }));
      }
      if (target === 'position-section') {
        loadFinancialSchedules().catch((err) => toast(err.message, { tone: 'error' }));
      }
      if (target === 'cashflow-section') {
        loadFinancialSchedules().catch((err) => toast(err.message, { tone: 'error' }));
        loadCashflowCharts().catch((err) => toast(`Cash flow charts failed: ${err.message}`, { tone: 'error' }));
      }
    });
  });
}

function initializeApiBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get('apiBase') || params.get('api_base');
  const stored = window.localStorage.getItem('ecom-api-base');
  const globalBase = window.ECOM_API_BASE;

  if (override) {
    setApiBase(override.trim());
    window.localStorage.setItem('ecom-api-base', getApiBase());
    toast(`API base set from query: ${getApiBase()}`);
    return;
  }

  if (globalBase) {
    setApiBase(String(globalBase).trim());
    return;
  }

  if (stored) {
    setApiBase(stored);
    return;
  }

  if (window.location.origin && window.location.origin.startsWith('http')) {
    const { hostname, port } = window.location;
    if ((hostname === 'localhost' || hostname === '127.0.0.1') && port && port !== '8000') {
      setApiBase('http://localhost:8000');
      return;
    }
    setApiBase(window.location.origin);
    return;
  }

  setApiBase('http://localhost:8000');
}

function wireInputPage() {
  qs('#refresh-file').addEventListener('click', () => refreshWorkbook('Load Existing'));
  qs('#start-new').addEventListener('click', async () => {
    try {
      await refreshWorkbook('Start New');
      toast('New workbook initialized.', { tone: 'info' });
    } catch (err) {
      toast(err.message, { tone: 'error' });
    }
  });

  const uploadForm = qs('#upload-form');
  uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const fileInput = qs('#assumption-file');
    if (!fileInput.files || fileInput.files.length === 0) {
      toast('Please choose an Excel workbook to upload.', { tone: 'error' });
      return;
    }
    try {
      const formData = new FormData(uploadForm);
      await apiRequest('/upload_file', { method: 'POST', body: formData });
      toast('Workbook uploaded and processed.');
      fileInput.value = '';
      await refreshWorkbook('Load Existing');
    } catch (err) {
      toast(`Upload failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#add-assumption-row').addEventListener('click', () => {
    if (state.assumptionColumns.length === 0) {
      state.assumptionColumns = getDefaultAssumptionColumns();
    }
    let nextYear = new Date().getFullYear();
    if (state.assumptions.length) {
      const sortedYears = state.assumptions
        .map((row) => Number(row[YEAR_COLUMN]))
        .filter((num) => Number.isFinite(num))
        .sort((a, b) => a - b);
      if (sortedYears.length) {
        nextYear = sortedYears[sortedYears.length - 1] + 1;
      }
    }
    const newRow = {};
    state.assumptionColumns.forEach((col) => {
      newRow[col] = col === YEAR_COLUMN ? nextYear : null;
    });
    state.assumptions.push(newRow);
    state.assumptions.sort((a, b) => Number(a[YEAR_COLUMN] || 0) - Number(b[YEAR_COLUMN] || 0));
    renderAssumptionTables();
  });

  qs('#reset-assumptions').addEventListener('click', () => {
    state.assumptions = JSON.parse(JSON.stringify(state.originalAssumptions));
    normalizeAssumptionRows();
    renderAssumptionTables();
  });

  qs('#save-assumptions').addEventListener('click', async () => {
    const collected = collectAssumptionsFromTables();
    if (!collected.length) {
      toast('Add at least one year of assumptions before saving.', { tone: 'error' });
      return;
    }
    try {
      state.assumptions = collected.map((row) => ({ ...row }));
      normalizeAssumptionRows();
      const body = state.assumptions.map((row) => {
        const entry = {};
        state.assumptionColumns.forEach((column) => {
          entry[column] = row[column] ?? null;
        });
        return entry;
      });
      const response = await apiPost('/save_assumptions', body);
      toast(response.message || 'Assumptions saved.');
      await refreshWorkbook('Load Existing');
    } catch (err) {
      toast(`Save failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#filter-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const body = Object.fromEntries(formData.entries());
    Object.keys(body).forEach((key) => {
      if (body[key] === '') delete body[key];
      else body[key] = Number(body[key]);
    });
    try {
      const response = await apiPost('/filter_time_period', body);
      toast(response.message || 'Filter applied.');
      await loadKeyMetrics();
      await loadFinancialSchedules();
    } catch (err) {
      toast(`Filter failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#base-analysis-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const body = Object.fromEntries(formData.entries());
    Object.keys(body).forEach((key) => {
      body[key] = Number(body[key]);
    });
    try {
      const response = await apiPost('/run_base_analysis', body);
      qs('#base-analysis-result').textContent = JSON.stringify(response, null, 2);
      toast('Base analysis completed.');
      await loadKeyMetrics();
      await loadFinancialSchedules();
    } catch (err) {
      toast(`Base analysis failed: ${err.message}`, { tone: 'error' });
    }
  });
}

async function refreshWorkbook(action) {
  try {
    const params = new URLSearchParams({ action });
    const response = await apiGet(`/file_action?${params.toString()}`);
    if (response?.data) {
      const rows = Object.values(response.data);
      state.assumptions = rows.map((row) => ({ ...row }));
      state.originalAssumptions = JSON.parse(JSON.stringify(state.assumptions));
      normalizeAssumptionRows();
      renderAssumptionTables();
      qs('#file-status').textContent = response.exists
        ? `Loaded ${response.filename || 'financial_assumptions.xlsx'}`
        : 'Workbook not found yet. Use Start New or upload a file.';
    } else {
      state.assumptions = [];
      state.assumptionColumns = getDefaultAssumptionColumns();
      renderAssumptionTables();
      qs('#file-status').textContent = 'No data returned. Upload a workbook to begin.';
    }
  } catch (err) {
    qs('#file-status').textContent = `Error: ${err.message}`;
    throw err;
  }
}

function getDefaultAssumptionColumns() {
  const ordered = [];
  ASSUMPTION_GROUPS.forEach((group) => {
    group.columns.forEach((column) => {
      if (!ordered.includes(column)) {
        ordered.push(column);
      }
    });
  });
  if (!ordered.includes(YEAR_COLUMN)) {
    ordered.unshift(YEAR_COLUMN);
  }
  return ordered;
}

function normalizeAssumptionRows() {
  const baseColumns = getDefaultAssumptionColumns();
  const columnSet = new Set(baseColumns);
  state.assumptions.forEach((row) => {
    Object.keys(row).forEach((column) => columnSet.add(column));
  });
  const extras = Array.from(columnSet).filter((column) => !baseColumns.includes(column)).sort((a, b) => a.localeCompare(b));
  state.assumptionColumns = [...baseColumns, ...extras].filter((column, index, arr) => arr.indexOf(column) === index);
  if (!state.assumptionColumns.includes(YEAR_COLUMN)) {
    state.assumptionColumns.unshift(YEAR_COLUMN);
  }
  state.assumptions.forEach((row) => {
    state.assumptionColumns.forEach((column) => {
      if (column === YEAR_COLUMN) return;
      if (!(column in row)) {
        row[column] = null;
      }
    });
  });
  state.assumptions.sort((a, b) => Number(a[YEAR_COLUMN] || 0) - Number(b[YEAR_COLUMN] || 0));
}

function renderAssumptionTables() {
  const container = qs('#assumption-groups');
  if (!container) return;
  clear(container);
  if (!state.assumptions.length) {
    container.append(
      el('p', {
        className: 'status',
        textContent: 'No assumptions loaded yet. Use the controls above to load or create a workbook.',
      })
    );
    return;
  }

  ASSUMPTION_GROUPS.forEach((group) => {
    const availableColumns = group.columns.filter((column) => state.assumptionColumns.includes(column));
    if (availableColumns.length <= 1) return;

    const rows = state.assumptions.map((row, index) => {
      const shaped = { __rowIndex: index };
      availableColumns.forEach((column) => {
        shaped[column] = row[column] ?? null;
      });
      return shaped;
    });

    const table = buildTable({ columns: availableColumns, rows, editable: true });
    table.dataset.columns = JSON.stringify(availableColumns);
    table.dataset.groupId = group.id;
    table.classList.add('assumption-table');
    table.addEventListener('input', handleAssumptionCellInput);

    container.append(
      el('section', {
        className: 'assumption-group',
        children: [
          el('header', {
            className: 'assumption-group__header',
            children: [
              el('h4', { textContent: group.title }),
              group.description
                ? el('p', {
                    className: 'assumption-group__description',
                    textContent: group.description,
                  })
                : null,
            ],
          }),
          el('div', {
            className: 'table-wrapper',
            children: [table],
          }),
        ],
      })
    );
  });
}

function handleAssumptionCellInput(event) {
  const cell = event.target.closest('td');
  if (!cell) return;
  const column = cell.dataset.column;
  if (!column || column === YEAR_COLUMN) return;
  const rowIndex = Number(cell.dataset.rowIndex);
  if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= state.assumptions.length) return;
  state.assumptions[rowIndex][column] = coerceValue(cell.textContent);
}

function collectAssumptionsFromTables() {
  const tables = qsa('#assumption-groups table');
  if (!tables.length) {
    return [];
  }
  const merged = new Map();
  tables.forEach((table) => {
    let columns = [];
    try {
      columns = JSON.parse(table.dataset.columns || '[]');
    } catch (err) {
      columns = [];
    }
    if (!Array.isArray(columns) || columns.length === 0) {
      return;
    }
    const rows = tableToData(table, columns);
    rows.forEach((row) => {
      const yearValue = row[YEAR_COLUMN];
      if (yearValue == null || Number.isNaN(Number(yearValue))) {
        return;
      }
      const year = Number(yearValue);
      if (!merged.has(year)) {
        merged.set(year, { [YEAR_COLUMN]: year });
      }
      const target = merged.get(year);
      Object.entries(row).forEach(([key, value]) => {
        if (key === YEAR_COLUMN) return;
        target[key] = value ?? null;
      });
    });
  });

  const result = Array.from(merged.values());
  result.forEach((row) => {
    state.assumptionColumns.forEach((column) => {
      if (!(column in row)) {
        row[column] = null;
      }
    });
  });
  result.sort((a, b) => Number(a[YEAR_COLUMN] || 0) - Number(b[YEAR_COLUMN] || 0));
  return result;
}

function wireMetricsPage() {
  const refreshMetricsBtn = qs('#refresh-key-metrics');
  if (refreshMetricsBtn) {
    refreshMetricsBtn.addEventListener('click', () => {
      loadKeyMetrics().catch((err) => toast(err.message, { tone: 'error' }));
    });
  }

  const refreshChartsBtn = qs('#refresh-charts');
  if (refreshChartsBtn) {
    refreshChartsBtn.addEventListener('click', () => {
      loadCharts().catch((err) => toast(err.message, { tone: 'error' }));
    });
  }

  const scenarioForm = qs('#scenario-form');
  if (scenarioForm) {
    scenarioForm.addEventListener('submit', handleScenarioSubmit);
  }

  const scenarioSelect = qs('#scenario-type');
  if (scenarioSelect) {
    scenarioSelect.value = state.scenario.type;
    scenarioSelect.addEventListener('change', (event) => {
      state.scenario.type = event.target.value;
      loadScenarioDefaults(state.scenario.type, { silent: true }).catch(() => {});
    });
  }

  const defaultsBtn = qs('#scenario-load-defaults');
  if (defaultsBtn) {
    defaultsBtn.addEventListener('click', () => {
      loadScenarioDefaults(state.scenario.type, { silent: false }).catch(() => {});
    });
  }

  const implicationsBtn = qs('#refresh-implications');
  if (implicationsBtn) {
    implicationsBtn.addEventListener('click', () => {
      loadImplications();
    });
  }
}

function wirePerformancePage() {
  const refreshBtn = qs('#refresh-performance-charts');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      loadPerformanceCharts().catch((err) => toast(`Performance charts failed: ${err.message}`, { tone: 'error' }));
    });
  }
}

function wireCashflowPage() {
  const refreshBtn = qs('#refresh-cashflow-charts');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      loadCashflowCharts().catch((err) => toast(`Cash flow charts failed: ${err.message}`, { tone: 'error' }));
    });
  }
}

async function loadKeyMetrics() {
  await Promise.all([loadSummaryMetrics(), loadOperationalMetrics(), loadValuation(), loadScenarioMetrics(), loadCharts()]);
  await loadImplications();
}

async function loadSummaryMetrics() {
  const container = qs('#key-metrics-container');
  try {
    const data = await apiGet('/display_metrics_summary_of_analysis');
    if (!Array.isArray(data) || data.length === 0) {
      clear(container);
      container.append(el('p', { className: 'status', textContent: 'No summary metrics available.' }));
      return;
    }
    const columns = Object.keys(data[0]);
    const table = buildTable({ columns, rows: data });
    mountTable(container, table);
  } catch (err) {
    clear(container);
    container.append(el('p', { className: 'status', textContent: `Failed to load metrics: ${err.message}` }));
    throw err;
  }
}

async function loadOperationalMetrics() {
  const container = qs('#operational-metrics-container');
  try {
    const data = await apiGet('/operational_metrics');
    renderMetricGrid(container, data.metrics || []);
  } catch (err) {
    clear(container);
    container.append(el('p', { className: 'status', textContent: `Failed to load operational metrics: ${err.message}` }));
    throw err;
  }
}

async function loadValuation() {
  const container = qs('#valuation-container');
  try {
    const data = await apiGet('/dcf_valuation');
    renderKeyValueTable(container, {
      'Enterprise Value ($M)': `${Number(data.enterprise_value_m).toFixed(1)}`,
      'Equity Value ($M)': `${Number(data.equity_value_m).toFixed(1)}`,
    });
  } catch (err) {
    clear(container);
    container.append(el('p', { className: 'status', textContent: `Failed to load valuation: ${err.message}` }));
    throw err;
  }
}

async function loadScenarioMetrics() {
  const container = qs('#scenario-metrics-container');
  try {
    const data = await apiGet('/display_metrics_scenario_analysis');
    const payload = data?.data;
    if (!payload) {
      clear(container);
      container.append(el('p', { className: 'status', textContent: 'No scenario metrics available.' }));
      return;
    }
    if (Array.isArray(payload)) {
      const columns = Object.keys(payload[0]);
      const table = buildTable({ columns, rows: payload });
      mountTable(container, table);
    } else if (typeof payload === 'object') {
      renderKeyValueTable(container, payload);
    } else {
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(payload, null, 2) }));
    }
  } catch (err) {
    clear(container);
    container.append(el('p', { className: 'status', textContent: `Failed to load scenario metrics: ${err.message}` }));
    throw err;
  }
}

function renderScenarioParameters() {
  const container = qs('#scenario-params');
  if (!container) return;
  clear(container);
  const params = state.scenario.params || getScenarioDefaults(state.scenario.type);
  Object.entries(params).forEach(([key, value]) => {
    const input = el('input', {
      attrs: {
        type: 'number',
        step: '0.01',
        name: key,
        value: value ?? '',
      },
    });
    input.addEventListener('input', (event) => {
      const raw = event.target.value;
      if (raw === '') {
        state.scenario.params[key] = null;
        return;
      }
      const numeric = Number(raw);
      if (Number.isFinite(numeric)) {
        state.scenario.params[key] = numeric;
      }
    });
    container.append(
      el('label', {
        className: 'stack',
        children: [
          el('span', { textContent: formatScenarioLabel(key) }),
          input,
        ],
      })
    );
  });
}

async function loadScenarioDefaults(type, { silent = false } = {}) {
  const fallback = getScenarioDefaults(type);
  try {
    const response = await apiGet(`/get_scenario_parameters/${encodeURIComponent(type)}`);
    const parameters = (response && (response.parameters || response)) || null;
    if (parameters && typeof parameters === 'object') {
      state.scenario.params = { ...fallback, ...parameters };
      if (!silent) {
        toast('Scenario defaults loaded.', { tone: 'success' });
      }
    } else {
      state.scenario.params = fallback;
      if (!silent) {
        toast('Using built-in scenario defaults.', { tone: 'info' });
      }
    }
  } catch (err) {
    state.scenario.params = fallback;
    const message = (err.message || '').toLowerCase();
    if (!silent) {
      if (message.includes('not authenticated')) {
        toast('Sign in to load saved scenario defaults. Local defaults applied instead.', { tone: 'info' });
      } else {
        toast(`Failed to load scenario defaults: ${err.message}`, { tone: 'error' });
      }
    }
  }
  renderScenarioParameters();
}

function collectScenarioParams() {
  const params = { ...getScenarioDefaults(state.scenario.type), ...(state.scenario.params || {}) };
  const container = qs('#scenario-params');
  if (container) {
    qsa('input', container).forEach((input) => {
      const name = input.name;
      const raw = input.value;
      if (!name || raw === '') return;
      const numeric = Number(raw);
      if (Number.isFinite(numeric)) {
        params[name] = numeric;
      }
    });
  }
  return params;
}

async function handleScenarioSubmit(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {
    scenario_type: state.scenario.type,
    discount_rate: Number(form.discount_rate.value),
    tax_rate: Number(form.tax_rate.value),
    inflation_rate: Number(form.inflation_rate.value),
    direct_labor_rate_increase: Number(form.direct_labor_rate_increase.value),
    scenario_params: collectScenarioParams(),
  };
  const resultContainer = qs('#scenario-result');
  try {
    const response = await apiPost('/select_scenario', payload);
    state.scenarioResult = response.scenario_df || null;
    renderScenarioResult(state.scenarioResult, response.message);
    toast(response.message || 'Scenario updated.');
    await loadKeyMetrics().catch(() => {});
    await loadFinancialSchedules().catch(() => {});
    await loadPerformanceCharts().catch(() => {});
    await loadCashflowCharts().catch(() => {});
  } catch (err) {
    if (resultContainer) {
      clear(resultContainer);
      resultContainer.append(
        el('p', {
          className: 'status',
          textContent: `Scenario update failed: ${err.message}`,
        })
      );
    }
    toast(`Scenario update failed: ${err.message}`, { tone: 'error' });
  }
}

function renderScenarioResult(rows, message) {
  const container = qs('#scenario-result');
  if (!container) return;
  clear(container);
  if (message) {
    container.append(el('p', { className: 'status', textContent: message }));
  }
  if (Array.isArray(rows) && rows.length) {
    const columns = Object.keys(rows[0]);
    const table = buildTable({ columns, rows });
    container.append(table);
  } else if (!message) {
    container.append(el('p', { className: 'status', textContent: 'Submit the form to generate scenario outputs.' }));
  }
}

function formatScenarioLabel(key) {
  return SCENARIO_PARAM_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

async function loadImplications() {
  const container = qs('#implications-container');
  if (!container) return;
  try {
    const response = await apiGet('/key_implications');
    const items = response?.implications || response;
    clear(container);
    if (Array.isArray(items) && items.length) {
      container.append(
        el('ul', {
          className: 'implications-list',
          children: items.map((item) => el('li', { textContent: item })),
        })
      );
    } else if (typeof items === 'string') {
      container.append(el('p', { className: 'status', textContent: items }));
    } else {
      container.append(el('p', { className: 'status', textContent: 'No implications available.' }));
    }
  } catch (err) {
    clear(container);
    const message = (err.message || '').toLowerCase().includes('not authenticated')
      ? 'Sign in to view qualitative implications.'
      : `Failed to load implications: ${err.message}`;
    container.append(el('p', { className: 'status', textContent: message }));
  }
}

async function loadCharts() {
  try {
    const [revenue, traffic, profitability, cashflow] = await Promise.all([
      apiGet('/revenue_chart_data'),
      apiGet('/traffic_chart_data'),
      apiGet('/profitability_chart_data'),
      apiGet('/cashflow_forecast_chart_data'),
    ]);
    renderPlotlyChart(
      'revenue-chart',
      buildRevenueTraces(revenue?.revenue_chart_data || revenue?.data || revenue),
      {
        barmode: 'group',
        yaxis: { title: 'Net revenue', tickprefix: '$', separatethousands: true },
        yaxis2: { title: 'Margins', overlaying: 'y', side: 'right', tickformat: '.0%' },
        legend: { orientation: 'h', y: -0.2 },
      }
    );
    renderPlotlyChart(
      'traffic-chart',
      buildTrafficTraces(traffic?.traffic_chart_data || traffic?.data || traffic),
      {
        barmode: 'group',
        yaxis: { title: 'LTV & CAC', tickprefix: '$', separatethousands: true },
        yaxis2: { title: 'LTV to CAC ratio', overlaying: 'y', side: 'right' },
        legend: { orientation: 'h', y: -0.2 },
      }
    );
    renderPlotlyChart(
      'profitability-chart',
      buildProfitabilityTraces(profitability?.profitability_chart_data || profitability?.data || profitability),
      { yaxis: { title: 'Closing cash balance', tickprefix: '$', separatethousands: true } }
    );
    renderPlotlyChart(
      'cashflow-forecast-chart',
      buildCashflowForecastTraces(cashflow?.cashflow_forecast_chart_data || cashflow?.data || cashflow),
      { barmode: 'group', yaxis: { title: 'Cash flow', tickprefix: '$', separatethousands: true } }
    );
  } catch (err) {
    toast(`Failed to load charts: ${err.message}`, { tone: 'error' });
    throw err;
  }
}

async function loadFinancialSchedules() {
  const types = [
    'Income Statement',
    'Customer Metrics',
    'Balance Sheet',
    'Capital Assets',
    'Cash Flow Statement',
    'Debt Payment Schedule',
  ];
  const params = new URLSearchParams();
  types.forEach((type) => params.append('schedules', type));
  const response = await apiGet(`/financial_schedules?${params.toString()}`);
  const scheduleMap = new Map();
  (response.schedules || []).forEach((item) => {
    scheduleMap.set(item.schedule, item.data);
  });
  renderScheduleTable(qs('#income-statement-container'), scheduleMap.get('Income Statement'));
  renderScheduleTable(qs('#customer-metrics-container'), scheduleMap.get('Customer Metrics'));
  renderScheduleTable(qs('#balance-sheet-container'), scheduleMap.get('Balance Sheet'));
  renderScheduleTable(qs('#capital-assets-container'), scheduleMap.get('Capital Assets'));
  renderScheduleTable(qs('#cashflow-container'), scheduleMap.get('Cash Flow Statement'));
  const debtData = scheduleMap.get('Debt Payment Schedule');
  renderScheduleTable(qs('#debt-schedule-container'), flattenDebtSchedule(debtData));
}

function flattenDebtSchedule(data) {
  if (!Array.isArray(data)) return [];
  const rows = [];
  data.forEach((yearItem) => {
    (yearItem.Schedules || []).forEach((debt) => {
      (debt.Schedule || []).forEach((entry) => {
        rows.push({
          Year: entry.Year,
          Debt_Name: debt.Debt_Name,
          Principal: entry.Principal,
          Interest: entry.Interest,
          Payment: entry.Payment,
        });
      });
    });
  });
  return rows;
}

function wireSensitivityPage() {
  qs('#top-rank-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const variables = formData.get('variables').split(',').map((v) => v.trim()).filter(Boolean);
    const payload = {
      variables_to_test: variables,
      change_percentage: Number(formData.get('change_percentage')),
      discount_rate: Number(formData.get('discount_rate')),
    };
    try {
      const response = await apiPost('/top_rank_sensitivity', payload);
      const container = qs('#top-rank-results');
      clear(container);
      const rows = (response.sensitivity_results || []).map((row) => ({
        Variable: row.variable,
        Direction: row.direction,
        'Net Income Δ': row.net_income_change,
        'EBITDA Δ': row.ebitda_change,
        'Net Cash Flow Δ': row.net_cash_flow_change,
        'Equity Value Δ': row.equity_value_change,
      }));
      if (rows.length) {
        const table = buildTable({ columns: Object.keys(rows[0]), rows });
        mountTable(container, table);
      } else {
        container.append(el('p', { className: 'status', textContent: response.message || 'No results.' }));
      }
      if (response.sensitivity_insights?.length) {
        container.append(
          el('ul', {
            children: response.sensitivity_insights.map((insight) => el('li', { textContent: insight })),
          })
        );
      }
    } catch (err) {
      toast(`Sensitivity failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#what-if-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const body = {
      variable: form.variable.value,
      year: Number(form.year.value),
      multiplier: Number(form.multiplier.value),
    };
    try {
      const response = await apiPost('/what_if', body);
      const container = qs('#what-if-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('What-if analysis complete.');
    } catch (err) {
      toast(`What-if failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#goal-seek-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = {
      target_metric: form.metric.value,
      target_year: Number(form.year.value),
      target_value: Number(form.target.value),
      driver: form.driver.value,
    };
    try {
      const response = await apiPost('/goal_seek', payload);
      const container = qs('#goal-seek-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Goal seek finished.');
    } catch (err) {
      toast(`Goal seek failed: ${err.message}`, { tone: 'error' });
    }
  });
}

async function loadPerformanceCharts() {
  const chartIds = ['waterfall-chart', 'breakeven-chart', 'consideration-chart', 'margin-safety-chart', 'margin-trends-chart'];
  try {
    const [waterfall, breakeven, consideration, marginSafety, marginTrends] = await Promise.all([
      apiGet('/waterfall_chart_data'),
      apiGet('/breakeven_chart_data'),
      apiGet('/consideration_chart_data'),
      apiGet('/margin_safety_chart'),
      apiGet('/profitability_margin_trends_chart_data'),
    ]);
    renderPlotlyChart('waterfall-chart', buildWaterfallTrace(waterfall?.data || waterfall), {
      showlegend: false,
    });
    renderPlotlyChart('breakeven-chart', buildBreakevenTraces(breakeven?.data || breakeven), {
      yaxis: { title: 'Revenue', tickprefix: '$', separatethousands: true },
      legend: { orientation: 'h', y: -0.2 },
    });
    renderPlotlyChart('consideration-chart', buildConsiderationTraces(consideration?.data || consideration), {
      yaxis: { title: 'Consideration rates', tickformat: '.0%' },
      legend: { orientation: 'h', y: -0.2 },
    });
    renderPlotlyChart('margin-safety-chart', buildMarginSafetyTraces(marginSafety?.data || marginSafety), {
      yaxis: { title: 'Margin of safety ($)', tickprefix: '$', separatethousands: true },
      yaxis2: { title: 'Margin of safety %', overlaying: 'y', side: 'right', tickformat: '.0%' },
      legend: { orientation: 'h', y: -0.2 },
    });
    renderPlotlyChart('margin-trends-chart', buildMarginTrendTraces(marginTrends?.data || marginTrends), {
      yaxis: { title: 'Margin %', tickformat: '.0%' },
      legend: { orientation: 'h', y: -0.2 },
    });
  } catch (err) {
    showChartError(chartIds, `Failed to load performance charts: ${err.message}`);
    throw err;
  }
}

async function loadCashflowCharts() {
  const chartIds = ['cashflow-forecast-detailed', 'dcf-summary-chart'];
  try {
    const [cashflow, dcfSummary] = await Promise.all([
      apiGet('/cashflow_forecast_chart_data'),
      apiGet('/dcf_summary_chart_data'),
    ]);
    renderPlotlyChart(
      'cashflow-forecast-detailed',
      buildCashflowForecastTraces(cashflow?.cashflow_forecast_chart_data || cashflow?.data || cashflow),
      { barmode: 'group', yaxis: { title: 'Cash flow', tickprefix: '$', separatethousands: true } }
    );
    renderPlotlyChart('dcf-summary-chart', buildWaterfallTrace(dcfSummary?.data || dcfSummary), {
      showlegend: false,
    });
  } catch (err) {
    showChartError(chartIds, `Failed to load cash flow charts: ${err.message}`);
    throw err;
  }
}

function showChartError(ids, message) {
  ids.forEach((id) => {
    const node = document.getElementById(id);
    if (node) {
      node.innerHTML = `<p class="status">${message}</p>`;
    }
  });
}

function buildRevenueTraces(data = {}) {
  const years = ensureArray(data.years);
  const traces = [];
  const revenue = toNumberSeries(data.net_revenue);
  const grossMargin = toPercentSeries(data.gross_margin);
  const ebitdaMargin = toPercentSeries(data.ebitda_margin);
  if (years.length && revenue.some(isFiniteNumber)) {
    traces.push({
      type: 'bar',
      name: 'Net Revenue',
      x: years,
      y: revenue,
      marker: { color: '#2563eb' },
    });
  }
  if (years.length && grossMargin.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Gross Margin',
      x: years,
      y: grossMargin,
      yaxis: 'y2',
      marker: { color: '#0ea5e9' },
      line: { width: 3 },
    });
  }
  if (years.length && ebitdaMargin.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'EBITDA Margin',
      x: years,
      y: ebitdaMargin,
      yaxis: 'y2',
      marker: { color: '#a855f7' },
      line: { dash: 'dot', width: 3 },
    });
  }
  return traces;
}

function buildTrafficTraces(data = {}) {
  const years = ensureArray(data.years);
  const ltv = toNumberSeries(data.ltv);
  const cac = toNumberSeries(data.cac);
  const ratio = toNumberSeries(data.ltv_cac_ratio);
  const traces = [];
  if (years.length && ltv.some(isFiniteNumber)) {
    traces.push({ type: 'bar', name: 'LTV', x: years, y: ltv, marker: { color: '#22c55e' } });
  }
  if (years.length && cac.some(isFiniteNumber)) {
    traces.push({ type: 'bar', name: 'CAC', x: years, y: cac, marker: { color: '#ef4444' } });
  }
  if (years.length && ratio.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'LTV / CAC Ratio',
      x: years,
      y: ratio,
      yaxis: 'y2',
      marker: { color: '#6366f1' },
      line: { width: 3 },
    });
  }
  return traces;
}

function buildProfitabilityTraces(data = {}) {
  const years = ensureArray(data.years);
  const balance = toNumberSeries(data.closing_cash_balance);
  if (!years.length || !balance.some(isFiniteNumber)) return [];
  return [
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Closing Cash Balance',
      x: years,
      y: balance,
      line: { color: '#f97316', width: 3 },
      marker: { color: '#f97316' },
    },
  ];
}

function buildCashflowForecastTraces(data = {}) {
  const years = ensureArray(data.years);
  const operations = toNumberSeries(data.cash_from_operations);
  const investing = toNumberSeries(data.cash_from_investing);
  const net = toNumberSeries(data.net_cash_flow);
  const traces = [];
  if (years.length && operations.some(isFiniteNumber)) {
    traces.push({ type: 'bar', name: 'Operations', x: years, y: operations, marker: { color: '#22c55e' } });
  }
  if (years.length && investing.some(isFiniteNumber)) {
    traces.push({ type: 'bar', name: 'Investing', x: years, y: investing, marker: { color: '#f59e0b' } });
  }
  if (years.length && net.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Net Cash Flow',
      x: years,
      y: net,
      marker: { color: '#0ea5e9' },
      line: { width: 3 },
    });
  }
  return traces;
}

function buildWaterfallTrace(data = {}) {
  const categories = ensureArray(data.categories);
  const values = toNumberSeries(data.values);
  const measures = ensureArray(data.measures);
  if (!categories.length || !values.some(isFiniteNumber)) return [];
  return [
    {
      type: 'waterfall',
      name: data.title || 'Bridge',
      orientation: 'v',
      x: categories,
      measure: measures.length === categories.length ? measures : undefined,
      y: values,
      increasing: { marker: { color: '#22c55e' } },
      decreasing: { marker: { color: '#ef4444' } },
      totals: { marker: { color: '#6366f1' } },
    },
  ];
}

function buildBreakevenTraces(data = {}) {
  const years = ensureArray(data.years);
  const breakEven = toNumberSeries(data.break_even_dollars);
  const actual = toNumberSeries(data.actual_sales);
  const traces = [];
  if (years.length && breakEven.some(isFiniteNumber)) {
    traces.push({ type: 'scatter', mode: 'lines+markers', name: 'Break-even', x: years, y: breakEven, line: { color: '#ef4444', width: 3 } });
  }
  if (years.length && actual.some(isFiniteNumber)) {
    traces.push({ type: 'scatter', mode: 'lines+markers', name: 'Actual Sales', x: years, y: actual, line: { color: '#22c55e', width: 3 } });
  }
  return traces;
}

function buildConsiderationTraces(data = {}) {
  const years = ensureArray(data.years);
  const consideration = toPercentSeries(data.weighted_consideration_rate);
  const conversion = toPercentSeries(data.consideration_to_conversion);
  const traces = [];
  if (years.length && consideration.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Weighted consideration rate',
      x: years,
      y: consideration,
      line: { color: '#0ea5e9', width: 3 },
    });
  }
  if (years.length && conversion.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Consideration to conversion',
      x: years,
      y: conversion,
      line: { color: '#6366f1', width: 3, dash: 'dot' },
    });
  }
  return traces;
}

function buildMarginSafetyTraces(data = {}) {
  const years = ensureArray(data.years);
  const dollars = toNumberSeries(data.margin_safety_dollars);
  const percent = toPercentSeries(data.margin_safety_percentage);
  const traces = [];
  if (years.length && dollars.some(isFiniteNumber)) {
    traces.push({ type: 'bar', name: 'Margin of safety ($)', x: years, y: dollars, marker: { color: '#14b8a6' } });
  }
  if (years.length && percent.some(isFiniteNumber)) {
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Margin of safety %',
      x: years,
      y: percent,
      yaxis: 'y2',
      line: { color: '#0ea5e9', width: 3 },
    });
  }
  return traces;
}

function buildMarginTrendTraces(data = {}) {
  const years = ensureArray(data.years);
  const gross = toPercentSeries(data.gross_margin);
  const ebitda = toPercentSeries(data.ebitda_margin);
  const net = toPercentSeries(data.net_profit_margin);
  const traces = [];
  if (years.length && gross.some(isFiniteNumber)) {
    traces.push({ type: 'scatter', mode: 'lines+markers', name: 'Gross margin', x: years, y: gross, line: { color: '#22c55e', width: 3 } });
  }
  if (years.length && ebitda.some(isFiniteNumber)) {
    traces.push({ type: 'scatter', mode: 'lines+markers', name: 'EBITDA margin', x: years, y: ebitda, line: { color: '#6366f1', width: 3 } });
  }
  if (years.length && net.some(isFiniteNumber)) {
    traces.push({ type: 'scatter', mode: 'lines+markers', name: 'Net profit margin', x: years, y: net, line: { color: '#f97316', width: 3 } });
  }
  return traces;
}

function ensureArray(value) {
  return Array.isArray(value) ? value : [];
}

function toNumberSeries(values) {
  return ensureArray(values).map((val) => (typeof val === 'number' && Number.isFinite(val) ? val : null));
}

function toPercentSeries(values) {
  return ensureArray(values).map((val) => {
    if (typeof val === 'number' && Number.isFinite(val)) {
      return val / 100;
    }
    return null;
  });
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function wireAdvancedPage() {
  const downloadBtn = qs('#download-report');
  const downloadStatus = qs('#download-status');
  if (downloadBtn) {
    downloadBtn.addEventListener('click', async () => {
      if (downloadStatus) {
        downloadStatus.textContent = 'Preparing download…';
      }
      try {
        const blob = await apiGet('/export_excel');
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'ecommerce_report.xlsx';
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        if (downloadStatus) {
          downloadStatus.textContent = 'Download ready.';
        }
        toast('Excel report downloaded.', { tone: 'success' });
      } catch (err) {
        if (downloadStatus) {
          downloadStatus.textContent = `Export failed: ${err.message}`;
        }
        toast(`Export failed: ${err.message}`, { tone: 'error' });
      }
    });
  }

  qs('#monte-carlo-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = Object.fromEntries(new FormData(form).entries());
    Object.keys(payload).forEach((key) => (payload[key] = Number(payload[key])));
    try {
      const response = await apiPost('/monte_carlo', payload);
      const container = qs('#monte-carlo-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Monte Carlo simulation completed.');
    } catch (err) {
      toast(`Monte Carlo failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#schedule-risk-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    let tasks;
    const tasksValue = form.tasks.value.trim();
    if (tasksValue) {
      try {
        tasks = JSON.parse(tasksValue);
      } catch (err) {
        toast('Tasks must be valid JSON.', { tone: 'error' });
        return;
      }
    } else {
      tasks = [];
    }
    const payload = {
      confidence_level: Number(form.confidence_level.value),
      tasks,
    };
    try {
      const response = await apiPost('/schedule_risk_analysis', payload);
      const container = qs('#schedule-risk-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Schedule risk analysis complete.');
    } catch (err) {
      toast(`Schedule risk failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#neural-tools-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = {
      traffic_increase_percentage: Number(form.traffic_increase_percentage.value),
    };
    try {
      const response = await apiPost('/neural_tools', payload);
      const container = qs('#neural-tools-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Neural tools forecast ready.');
    } catch (err) {
      toast(`Neural tools failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#precision-tree-btn').addEventListener('click', async () => {
    try {
      const response = await apiGet('/precision_tree');
      const container = qs('#precision-tree-results');
      clear(container);
      if (response.decision_tree_image) {
        const img = el('img', {
          attrs: {
            src: `data:image/png;base64,${response.decision_tree_image}`,
            alt: 'Precision tree analysis',
          },
        });
        container.append(img);
      }
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Decision analysis retrieved.');
    } catch (err) {
      toast(`Precision tree failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#stat-forecast-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = {
      forecast_years: Number(form.forecast_years.value),
      confidence_level: Number(form.confidence_level.value),
    };
    try {
      const response = await apiPost('/stat_tools_forecasting', payload);
      const container = qs('#stat-forecast-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Forecast complete.');
    } catch (err) {
      toast(`Statistical forecast failed: ${err.message}`, { tone: 'error' });
    }
  });

  qs('#evolver-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const payload = {
      budget_dict: { [form.budget_line.value]: Number(form.budget_amount.value) },
      forecast_years: Number(form.forecast_years.value),
    };
    try {
      const response = await apiPost('/evolver_optimization', payload);
      const container = qs('#evolver-results');
      clear(container);
      container.append(el('pre', { className: 'code', textContent: JSON.stringify(response, null, 2) }));
      toast('Optimization finished.');
    } catch (err) {
      toast(`Optimization failed: ${err.message}`, { tone: 'error' });
    }
  });
}

document.addEventListener('DOMContentLoaded', init);
