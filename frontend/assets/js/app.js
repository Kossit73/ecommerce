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
};

function init() {
  wireNavigation();
  wireApiBaseInput();
  wireInputPage();
  wireMetricsPage();
  wireSensitivityPage();
  wireAdvancedPage();
  setApiBase(qs('#api-base').value || window.localStorage.getItem('ecom-api-base') || '');
  if (getApiBase()) {
    qs('#api-base').value = getApiBase();
  }
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
      if (['performance-section', 'position-section', 'cashflow-section'].includes(target)) {
        loadFinancialSchedules().catch((err) => toast(err.message, { tone: 'error' }));
      }
    });
  });
}

function wireApiBaseInput() {
  const applyBtn = qs('#apply-api-base');
  const input = qs('#api-base');
  applyBtn.addEventListener('click', () => {
    setApiBase(input.value.trim());
    window.localStorage.setItem('ecom-api-base', getApiBase());
    toast(`API base set to ${getApiBase() || 'relative path'}`);
  });
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
  qs('#refresh-key-metrics').addEventListener('click', () => {
    loadKeyMetrics().catch((err) => toast(err.message, { tone: 'error' }));
  });
  qs('#refresh-charts').addEventListener('click', () => {
    loadCharts().catch((err) => toast(err.message, { tone: 'error' }));
  });
}

async function loadKeyMetrics() {
  await Promise.all([loadSummaryMetrics(), loadOperationalMetrics(), loadValuation(), loadScenarioMetrics(), loadCharts()]);
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

async function loadCharts() {
  try {
    const [revenue, traffic, profitability, cashflow] = await Promise.all([
      apiGet('/revenue_chart_data'),
      apiGet('/traffic_chart_data'),
      apiGet('/profitability_chart_data'),
      apiGet('/cashflow_forecast_chart_data'),
    ]);
    renderPlotlyChart('revenue-chart', revenue?.traces || revenue);
    renderPlotlyChart('traffic-chart', traffic?.traces || traffic);
    renderPlotlyChart('profitability-chart', profitability?.traces || profitability);
    renderPlotlyChart('cashflow-forecast-chart', cashflow?.traces || cashflow);
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

function wireAdvancedPage() {
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
