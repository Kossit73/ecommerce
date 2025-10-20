import { setApiBase, getApiBase, apiGet, apiPost, apiRequest } from './api.js';
import { qs, qsa, clear, el, toast, formatCurrency } from './dom.js';
import {
  buildTable,
  mountTable,
  tableToData,
  renderMetricGrid,
  renderKeyValueTable,
  renderScheduleTable,
} from './tables.js';
import { renderPlotlyChart } from './charts.js';

const state = {
  assumptions: [],
  originalAssumptions: [],
  assumptionColumns: [],
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
      state.assumptionColumns = ['Year'];
    }
    const yearColumn = state.assumptionColumns.find((col) => col.toLowerCase() === 'year');
    let nextYear = new Date().getFullYear();
    if (state.assumptions.length && yearColumn) {
      const sortedYears = state.assumptions
        .map((row) => Number(row[yearColumn]))
        .filter((num) => Number.isFinite(num))
        .sort((a, b) => a - b);
      if (sortedYears.length) {
        nextYear = sortedYears[sortedYears.length - 1] + 1;
      }
    }
    const newRow = {};
    state.assumptionColumns.forEach((col) => {
      newRow[col] = col === yearColumn ? nextYear : null;
    });
    state.assumptions.push(newRow);
    renderAssumptionsTable();
  });

  qs('#reset-assumptions').addEventListener('click', () => {
    state.assumptions = JSON.parse(JSON.stringify(state.originalAssumptions));
    renderAssumptionsTable();
  });

  qs('#save-assumptions').addEventListener('click', async () => {
    const wrapper = qs('#assumptions-table-wrapper');
    const table = wrapper.querySelector('table');
    if (!table) {
      toast('Nothing to save yet. Load or add assumptions first.', { tone: 'error' });
      return;
    }
    const payload = tableToData(table, state.assumptionColumns);
    if (!payload.length) {
      toast('Add at least one year of assumptions before saving.', { tone: 'error' });
      return;
    }
    try {
      const response = await apiPost('/save_assumptions', payload);
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
      const columnSet = new Set();
      state.assumptions.forEach((row) => {
        Object.keys(row).forEach((key) => columnSet.add(key));
      });
      state.assumptionColumns = Array.from(columnSet);
      state.assumptionColumns.sort((a, b) => {
        if (a === 'Year') return -1;
        if (b === 'Year') return 1;
        return a.localeCompare(b);
      });
      renderAssumptionsTable();
      qs('#file-status').textContent = response.exists
        ? `Loaded ${response.filename || 'financial_assumptions.xlsx'}`
        : 'Workbook not found yet. Use Start New or upload a file.';
    } else {
      state.assumptions = [];
      state.assumptionColumns = [];
      renderAssumptionsTable();
      qs('#file-status').textContent = 'No data returned. Upload a workbook to begin.';
    }
  } catch (err) {
    qs('#file-status').textContent = `Error: ${err.message}`;
    throw err;
  }
}

function renderAssumptionsTable() {
  const wrapper = qs('#assumptions-table-wrapper');
  if (!state.assumptions.length) {
    clear(wrapper);
    wrapper.append(
      el('p', {
        className: 'status',
        textContent: 'No assumptions loaded yet. Use the controls above to load or create a workbook.',
      })
    );
    return;
  }
  const table = buildTable({ columns: state.assumptionColumns, rows: state.assumptions, editable: true });
  mountTable(wrapper, table);
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
