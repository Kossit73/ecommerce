// src/pages/FinanceDashboard.js

import React, { useEffect, useState, useRef } from 'react';
import {
  Alert,
  Box,
  Card,
  CardContent,
  CircularProgress,
  Collapse,
  IconButton,
  Tabs,
  Tab,
  TextField,
  Typography,
  Button,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import Sidebar from '../components/Sidebar';
import TrafficRevenueCard from '../components/cards/TrafficRevenueCard';
import MarketingExpensesCard from '../components/cards/MarketingExpensesCard';
import OfficeRentBreakdownCard from '../components/cards/OfficeRentBreakdownCard';
import ProfessionalFeesCard from '../components/cards/ProfessionalFeesCard';
import DepreciationBreakdownCard from '../components/cards/DepreciationBreakdownCard';
import BalanceSheetCard from '../components/cards/BalanceSheetCard';
import DebtIssuedCard from '../components/cards/DebtIssuedCard';
import BestCaseScenarioCard from '../components/cards/BestCaseScenarioCard';
import WorstCaseScenarioCard from '../components/cards/WorstCaseScenarioCard';
import SummaryCard from '../components/cards/SummaryCard';
import CombinedPerformanceCard from '../components/cards/CombinedPerformanceCard';
import MetricsAndExportCard from '../components/cards/MetricsAndExportCard';
import FinancialSchedulesCard from '../components/cards/FinancialSchedulesCard';
import DCFValuationCard from '../components/cards/DCFValuationCard';
import DetailedAnalysisCard from '../components/cards/DetailedAnalysisCard';
import AdvancedDecisionToolsCard from '../components/cards/AdvancedDecisionToolsCard';

import { defaults } from './default.js';
import BASE_URL from '../config';

export function pick(obj = {}, keys = []) {
  const out = {};
  keys.forEach((k) => {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      out[k] = obj[k];
    }
  });
  return out;
}

const PAGE_TABS = [
  { key: 'inputs', label: 'Inputs & Assumptions' },
  { key: 'metrics', label: 'Key Financial Metrics' },
  { key: 'performance', label: 'Financial Performance' },
  { key: 'position', label: 'Financial Position' },
  { key: 'cashflow', label: 'Cash Flow Statement' },
  { key: 'sensitivity', label: 'Sensitivity Analysis' },
  { key: 'advanced', label: 'Advanced Analysis' },
];

export default function FinanceDashboard() {
  const currentYear = new Date().getFullYear();
  const nextId = useRef(1000);

  function makeInitialBlock(preferredYear = currentYear) {
    return {
      id: nextId.current++,
      year: preferredYear,
      traffic: { ...defaults.traffic },
      marketing: { ...defaults.marketing },
      balance: { ...defaults.balance },
      officeRentRows: defaults.officeRentRows.map((r) => ({ ...r, id: nextId.current++ })),
      professionalFeesRows: defaults.professionalFeesRows.map((f) => ({ ...f, id: nextId.current++ })),
      depreciationBreakdown: defaults.depreciationBreakdown.map((a) => ({ ...a, id: nextId.current++ })),
      debtIssued: defaults.debtIssued.map((d) => ({ ...d, id: nextId.current++ })),
      EquityRaised: defaults.balance['Equity Raised'],
      DividendsPaid: defaults.balance['Dividends Paid'],
    };
  }

  const [allYearInputs, setAllYearInputs] = useState([makeInitialBlock()]);
  const [openBlocks, setOpenBlocks] = useState({});
  const [fileExists, setFileExists] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  const [kpisData, setKpisData] = useState(null);
  const [loadingKpis, setLoadingKpis] = useState(false);
  const [errorKpis, setErrorKpis] = useState('');
  const [summaryRowsData, setSummaryRowsData] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [errorSummary, setErrorSummary] = useState('');
  const [startYear, setStartYear] = useState(null);
  const [endYear, setEndYear] = useState(null);
  const [selectedOption, setSelectedOption] = useState('startNew');

  const [discountRate, setDiscountRate] = useState(20);
  const [wacc, setWacc] = useState(10);
  const [growthRate, setGrowthRate] = useState(2);
  const [taxRate, setTaxRate] = useState(0);
  const [inflationRate, setInflationRate] = useState(0);
  const [laborRateIncrease, setLaborRateIncrease] = useState(0);
  const [analysisPeriod, setAnalysisPeriod] = useState();
  const [scenario, setScenario] = useState('Base Case');

  const [loadingCharts, setLoadingCharts] = useState(false);
  const [chartsError, setChartsError] = useState(null);
  const [allChartsData, setAllChartsData] = useState({
    revenue: null,
    traffic: null,
    profitability: null,
    breakEven: null,
    consideration: null,
    marginSafety: null,
    cashflow: null,
    profitMargins: null,
    waterfall: null,
  });

  const [bestCaseData, setBestCaseData] = useState({
    conversion_rate_mult: 1.2,
    aov_mult: 1.1,
    cogs_mult: 0.95,
    interest_mult: 0.99,
    labor_mult: 0.99,
    material_mult: 0.99,
    markdown_mult: 0.99,
    political_risk: 2,
    env_impact: 2,
  });
  const [worstCaseData, setWorstCaseData] = useState({
    conversion_rate_mult: 0.8,
    aov_mult: 0.9,
    political_risk: 4,
    env_impact: 4,
    cogs_mult: 1.05,
    interest_mult: 1.2,
    labor_mult: 1.2,
    material_mult: 1.2,
    markdown_mult: 1.1,
  });

  const [forecastYears, setForecastYears] = useState(10);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [saveError, setSaveError] = useState('');

  const [activePage, setActivePage] = useState('inputs');

  async function runInitialAnalysis() {
    const payload = {
      discount_rate: discountRate / 100,
      wacc: wacc / 100,
      direct_labor_rate_increase: laborRateIncrease / 100,
      tax_rate: taxRate / 100,
      inflation_rate: inflationRate / 100,
      perpetual_growth: growthRate / 100,
      normal_forecast_years: forecastYears,
    };

    try {
      const resp = await fetch(`${BASE_URL}/run_base_analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(await resp.text());
      await resp.json();
    } catch (err) {
      console.error('Initial analysis failed:', err);
    }
  }

  function nestYear(rec) {
    const rawYear = rec.Year ?? rec.year;
    const chosenYear = rawYear ?? currentYear;

    const trafficInputs = pick(rec, Object.keys(defaults.traffic));
    const marketingInputs = pick(rec, Object.keys(defaults.marketing));
    const balanceInputs = pick(rec, Object.keys(defaults.balance));

    const officeRentRows = Array.isArray(rec.officeRentRows)
      ? rec.officeRentRows.map((r) => ({
          id: nextId.current++,
          category: r.category ?? defaults.officeRentRows[0].category,
          squareMeters: parseFloat(r.squareMeters) || 0,
          costPerSQM: parseFloat(r.costPerSQM) || 0,
        }))
      : defaults.officeRentRows.map((r) => ({ ...r, id: nextId.current++ }));

    const professionalFeesRows = Array.isArray(rec.professionalFeesRows)
      ? rec.professionalFeesRows.map((f) => ({
          id: nextId.current++,
          name: f.name,
          Cost: parseFloat(f.Cost) || 0,
        }))
      : defaults.professionalFeesRows.map((f) => ({ ...f, id: nextId.current++ }));

    const depreciationBreakdown = Array.isArray(rec.depreciationBreakdown)
      ? rec.depreciationBreakdown.map((a) => ({
          id: nextId.current++,
          name: a.name,
          amount: parseFloat(a.amount) || 0,
          rate: parseFloat(a.rate) || 0,
        }))
      : defaults.depreciationBreakdown.map((a) => ({ ...a, id: nextId.current++ }));

    const debtIssued = Array.isArray(rec.debtIssued)
      ? rec.debtIssued.map((d) => ({
          id: nextId.current++,
          name: d.name,
          amount: parseFloat(d.amount) || 0,
          interestRate: parseFloat(d.interestRate) || 0,
          duration: parseFloat(d.duration) || 0,
        }))
      : defaults.debtIssued.map((d) => ({ ...d, id: nextId.current++ }));

    const EquityRaised = rec['Equity Raised'] ?? defaults.balance['Equity Raised'];
    const DividendsPaid = rec['Dividends Paid'] ?? defaults.balance['Dividends Paid'];

    return {
      id: nextId.current++,
      year: chosenYear,
      traffic: trafficInputs,
      marketing: marketingInputs,
      balance: balanceInputs,
      officeRentRows,
      professionalFeesRows,
      depreciationBreakdown,
      debtIssued,
      EquityRaised,
      DividendsPaid,
    };
  }

  useEffect(() => {
    (async () => {
      let didExist = false;
      try {
        const resp = await fetch(`${BASE_URL}/file_action?action=Load%20Existing`);
        const json = await resp.json();

        const forecastFromServer = parseInt(json.forecastYears, 10);
        if (!Number.isNaN(forecastFromServer)) {
          setForecastYears(forecastFromServer);
        }

        if (json.exists) {
          didExist = true;
          let yearsArray = [];
          if (json.data && Array.isArray(json.data.years_data)) {
            yearsArray = json.data.years_data;
          } else if (json.data && Array.isArray(json.data)) {
            yearsArray = json.data;
          } else if (Array.isArray(json.years_data)) {
            yearsArray = json.years_data;
          } else if (Array.isArray(json)) {
            yearsArray = json;
          } else if (json.data && typeof json.data === 'object') {
            yearsArray = Object.values(json.data);
          }

          const nested = yearsArray.map(nestYear);
          setAllYearInputs(nested);
          const lastYear = Math.max(...nested.map((item) => +item.year || 0));
          setStartYear(lastYear);
          const forecastYearsToUse = forecastFromServer || forecastYears;
          setEndYear(lastYear + forecastYearsToUse);
          setAnalysisPeriod([lastYear, lastYear + (forecastYearsToUse + 1)]);
        }
      } catch (err) {
        console.warn('Load Existing failed:', err);
      }

      setFileExists(didExist);
      await runInitialAnalysis();

      setLoadingKpis(true);
      setLoadingSummary(true);
      setLoadingCharts(true);
      setErrorKpis('');
      setErrorSummary('');
      setChartsError(null);

      const url = `${BASE_URL}/display_metrics_scenario_analysis?selected_scenario=${encodeURIComponent(
        scenario,
      )}`;

      try {
        const [
          kpiResp,
          summaryResp,
          revResp,
          trafResp,
          profResp,
          beResp,
          consResp,
          msResp,
          cfResp,
          pmResp,
          wfResp,
        ] = await Promise.all([
          fetch(url),
          fetch(`${BASE_URL}/display_metrics_summary_of_analysis`),
          fetch(`${BASE_URL}/revenue_chart_data`),
          fetch(`${BASE_URL}/traffic_chart_data`),
          fetch(`${BASE_URL}/profitability_chart_data`),
          fetch(`${BASE_URL}/breakeven_chart_data`),
          fetch(`${BASE_URL}/consideration_chart_data`),
          fetch(`${BASE_URL}/margin_safety_chart`),
          fetch(`${BASE_URL}/cashflow_forecast_chart_data`),
          fetch(`${BASE_URL}/profitability_margin_trends_chart_data`),
          fetch(`${BASE_URL}/waterfall_chart_data`),
        ]);

        if (!kpiResp.ok) {
          throw new Error(`KPI fetch failed: ${await kpiResp.text()}`);
        }
        const kpiJson = await kpiResp.json();
        if (kpiJson.status === 'success' && kpiJson.data) {
          setKpisData(kpiJson.data);
        } else {
          throw new Error('Unexpected KPI payload');
        }

        if (!summaryResp.ok) {
          throw new Error(`Summary fetch failed: ${await summaryResp.text()}`);
        }
        const summaryJson = await summaryResp.json();
        if (Array.isArray(summaryJson)) {
          setSummaryRowsData(summaryJson);
        } else {
          throw new Error('Unexpected summary payload');
        }

        const assertOk = async (resp, name) => {
          if (!resp.ok) {
            throw new Error(`${name} fetch failed: ${resp.statusText}`);
          }
          return resp.json();
        };

        const revJson = await assertOk(revResp, 'Revenue');
        const trafJson = await assertOk(trafResp, 'Traffic');
        const profJson = await assertOk(profResp, 'Profitability');
        const beJson = await assertOk(beResp, 'Break-Even');
        const consJson = await assertOk(consResp, 'Consideration');
        const msJson = await assertOk(msResp, 'Margin Safety');
        const cfJson = await assertOk(cfResp, 'Cash Flow');
        const pmJson = await assertOk(pmResp, 'Profit Margins');
        const wfJson = await assertOk(wfResp, 'Waterfall');

        setAllChartsData({
          revenue: revJson.revenue_chart_data,
          traffic: trafJson.traffic_chart_data,
          profitability: profJson.data,
          breakEven: beJson.data,
          consideration: consJson.data,
          marginSafety: msJson.data,
          cashflow: cfJson.data,
          profitMargins: pmJson.data,
          waterfall: wfJson.data,
        });
      } catch (err) {
        console.error('Error fetching KPI / summary / charts:', err);
        const message = err.message || 'Unknown error fetching dashboard data';
        if (message.includes('KPI')) {
          setErrorKpis(message);
        } else if (message.includes('Summary')) {
          setErrorSummary(message);
        } else {
          setChartsError(message);
        }
      } finally {
        setLoadingKpis(false);
        setLoadingSummary(false);
        setLoadingCharts(false);
        setInitialLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!fileExists) {
      return;
    }

    (async () => {
      if (scenario !== 'Base Case') {
        try {
          const resp = await fetch(`${BASE_URL}/select_scenario`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              scenario_type: scenario,
              scenario_params: scenario === 'Best Case' ? bestCaseData : worstCaseData,
              discount_rate: discountRate / 100,
              tax_rate: taxRate / 100,
              inflation_rate: inflationRate / 100,
              direct_labor_rate_increase: laborRateIncrease / 100,
            }),
          });
          if (!resp.ok) {
            throw new Error(`select_scenario failed: ${await resp.text()}`);
          }
        } catch (err) {
          console.error('Error calling select_scenario:', err);
          return;
        }
      }

      await runInitialAnalysis();

      try {
        const resp = await fetch(`${BASE_URL}/filter_time_period`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scenario_type: scenario,
            start_year: analysisPeriod?.[0],
            end_year: analysisPeriod?.[1],
            discount_rate: discountRate / 100,
            wacc: wacc / 100,
            perpetual_growth: growthRate / 100,
            tax_rate: taxRate / 100,
            inflation_rate: inflationRate / 100,
            direct_labor_rate_increase: laborRateIncrease / 100,
          }),
        });
        if (!resp.ok) {
          throw new Error(`filter_time_period failed: ${await resp.text()}`);
        }
      } catch (err) {
        console.error('Error calling filter_time_period:', err);
        return;
      }

      const url = `${BASE_URL}/display_metrics_scenario_analysis?selected_scenario=${encodeURIComponent(
        scenario,
      )}`;
      setLoadingKpis(true);
      setErrorKpis('');
      fetch(url)
        .then((r) => {
          if (!r.ok) return r.text().then((txt) => Promise.reject(txt));
          return r.json();
        })
        .then((data) => {
          if (data.status === 'success' && data.data) {
            setKpisData(data.data);
          } else {
            throw new Error('Unexpected KPI payload');
          }
        })
        .catch((err) => {
          console.error('Error fetching KPIs on period change:', err);
          setErrorKpis(err.toString());
        })
        .finally(() => setLoadingKpis(false));

      setLoadingSummary(true);
      setErrorSummary('');
      fetch(`${BASE_URL}/display_metrics_summary_of_analysis`)
        .then((r) => {
          if (!r.ok) return r.text().then((txt) => Promise.reject(txt));
          return r.json();
        })
        .then((data) => {
          if (Array.isArray(data)) {
            setSummaryRowsData(data);
          } else {
            throw new Error('Unexpected summary payload');
          }
        })
        .catch((err) => {
          console.error('Error fetching summary on period change:', err);
          setErrorSummary(err.toString());
        })
        .finally(() => setLoadingSummary(false));

      setLoadingCharts(true);
      setChartsError(null);

      try {
        const [
          revResp,
          trafResp,
          profResp,
          beResp,
          consResp,
          msResp,
          cfResp,
          pmResp,
          wfResp,
        ] = await Promise.all([
          fetch(`${BASE_URL}/revenue_chart_data`),
          fetch(`${BASE_URL}/traffic_chart_data`),
          fetch(`${BASE_URL}/profitability_chart_data`),
          fetch(`${BASE_URL}/breakeven_chart_data`),
          fetch(`${BASE_URL}/consideration_chart_data`),
          fetch(`${BASE_URL}/margin_safety_chart`),
          fetch(`${BASE_URL}/cashflow_forecast_chart_data`),
          fetch(`${BASE_URL}/profitability_margin_trends_chart_data`),
          fetch(`${BASE_URL}/waterfall_chart_data`),
        ]);

        const assertOk = async (resp, name) => {
          if (!resp.ok) {
            throw new Error(`${name} fetch failed: ${resp.statusText}`);
          }
          return resp.json();
        };

        const revJson = await assertOk(revResp, 'Revenue');
        const trafJson = await assertOk(trafResp, 'Traffic');
        const profJson = await assertOk(profResp, 'Profitability');
        const beJson = await assertOk(beResp, 'Break-Even');
        const consJson = await assertOk(consResp, 'Consideration');
        const msJson = await assertOk(msResp, 'Margin Safety');
        const cfJson = await assertOk(cfResp, 'Cash Flow');
        const pmJson = await assertOk(pmResp, 'Profit Margins');
        const wfJson = await assertOk(wfResp, 'Waterfall');

        setAllChartsData({
          revenue: revJson.revenue_chart_data,
          traffic: trafJson.traffic_chart_data,
          profitability: profJson.data,
          breakEven: beJson.data,
          consideration: consJson.data,
          marginSafety: msJson.data,
          cashflow: cfJson.data,
          profitMargins: pmJson.data,
          waterfall: wfJson.data,
        });
      } catch (err) {
        console.error('Error fetching charts on period change:', err);
        setChartsError(err.message);
      } finally {
        setLoadingCharts(false);
      }
    })();
  }, [
    discountRate,
    wacc,
    growthRate,
    taxRate,
    inflationRate,
    laborRateIncrease,
    scenario,
    analysisPeriod,
    bestCaseData,
    worstCaseData,
    fileExists,
  ]);

  async function handleSaveAllData() {
    setSaving(true);
    setSaveError('');
    setSaveSuccess(false);

    const yearsData = allYearInputs.map((ent) => {
      const flat = { Year: ent.year };
      Object.assign(flat, ent.traffic);
      Object.assign(flat, ent.marketing);

      ent.officeRentRows.forEach((r) => {
        flat[`${r.category} Square Meters`] = r.squareMeters;
        flat[`${r.category} Cost per SQM`] = r.costPerSQM;
      });

      ent.professionalFeesRows.forEach((fee, idx) => {
        flat[`Professional_Fee_${idx + 1}_Name`] = fee.name;
        flat[`Professional_Fee_${idx + 1}_Cost`] = fee.Cost;
      });

      ent.depreciationBreakdown.forEach((asset, idx) => {
        flat[`Asset_${idx + 1}_Name`] = asset.name;
        flat[`Asset_${idx + 1}_Amount`] = asset.amount;
        flat[`Asset_${idx + 1}_Rate`] = asset.rate;
      });

      Object.assign(flat, ent.balance);

      ent.debtIssued.forEach((d, idx) => {
        flat[`Debt_${idx + 1}_Name`] = d.name;
        flat[`Debt_${idx + 1}_Amount`] = d.amount;
        flat[`Debt_${idx + 1}_Interest_Rate`] = d.interestRate;
        flat[`Debt_${idx + 1}_Duration`] = d.duration;
      });

      flat['Equity Raised'] = ent.EquityRaised;
      flat['Dividends Paid'] = ent.DividendsPaid;

      return flat;
    });

    const payload = {
      years_data: yearsData,
      forecastYears,
    };

    try {
      const resp = await fetch(`${BASE_URL}/save_assumptions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const json = await resp.json();
      if (json.status === 'success') {
        setSaveSuccess(true);
        setFileExists(true);
        setEndYear(json.last_year);
        setStartYear(json.last_year);
        const forecastYearsFromServer = json.forecastYears || forecastYears;
        setForecastYears(forecastYearsFromServer);
        const forecastHorizon = Number(forecastYearsFromServer) + 1;
        setAnalysisPeriod([json.last_year, json.last_year + forecastHorizon]);
        await runInitialAnalysis();
      } else {
        throw new Error(json.message || 'Unknown response from backend');
      }
    } catch (err) {
      console.error('Error saving assumptions:', err);
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const handleLoadExisting = async () => {
    setInitialLoading(true);
    try {
      const resp = await fetch(`${BASE_URL}/file_action?action=Load%20Existing`);
      const json = await resp.json();

      const forecastFromServer = parseInt(json.forecastYears, 10);
      if (!Number.isNaN(forecastFromServer)) {
        setForecastYears(forecastFromServer);
      }

      let yearsArray = [];
      if (json.data && Array.isArray(json.data.years_data)) {
        yearsArray = json.data.years_data;
      } else if (json.data && Array.isArray(json.data)) {
        yearsArray = json.data;
      } else if (Array.isArray(json.years_data)) {
        yearsArray = json.years_data;
      } else if (Array.isArray(json)) {
        yearsArray = json;
      }

      const nested = yearsArray.map(nestYear);
      setAllYearInputs(nested);
      setFileExists(true);
      if (nested.length > 0) {
        const lastYear = Math.max(...nested.map((item) => +item.year || 0));
        setStartYear(lastYear);
        const forecastYearsToUse = !Number.isNaN(forecastFromServer)
          ? forecastFromServer
          : forecastYears;
        setEndYear(lastYear + forecastYearsToUse);
        setAnalysisPeriod([lastYear, lastYear + forecastYearsToUse + 1]);
      }
      await runInitialAnalysis();
    } catch (err) {
      console.warn('Failed to reload existing file:', err);
    } finally {
      setInitialLoading(false);
    }
  };

  const handleStartNew = (year) => {
    setFileExists(false);
    setAllYearInputs([makeInitialBlock(year)]);
    setAnalysisPeriod([year, year + forecastYears + 1]);
  };

  const handleUploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      setInitialLoading(true);
      const resp = await fetch(`${BASE_URL}/upload_excel`, {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error(await resp.text());
      await handleLoadExisting();
    } catch (err) {
      console.error('Upload file failed:', err);
    } finally {
      setInitialLoading(false);
    }
  };

  const addYearBlock = () => {
    let computedYear = null;
    setAllYearInputs((prev) => {
      if (prev.length === 0) {
        const initialYear = startYear ?? currentYear;
        computedYear = initialYear;
        return [makeInitialBlock(initialYear)];
      }
      const lastBlock = prev[prev.length - 1];
      const nextYearValue = (lastBlock.year ?? currentYear) + 1;
      computedYear = nextYearValue;
      const cloned = structuredClone(lastBlock);
      cloned.id = Date.now();
      cloned.year = nextYearValue;
      cloned.officeRentRows = cloned.officeRentRows.map((r) => ({ ...r, id: nextId.current++ }));
      cloned.professionalFeesRows = cloned.professionalFeesRows.map((f) => ({ ...f, id: nextId.current++ }));
      cloned.depreciationBreakdown = cloned.depreciationBreakdown.map((a) => ({ ...a, id: nextId.current++ }));
      cloned.debtIssued = cloned.debtIssued.map((d) => ({ ...d, id: nextId.current++ }));
      return [...prev, cloned];
    });
    if (computedYear != null) {
      setAnalysisPeriod([computedYear, computedYear + forecastYears + 1]);
    }
  };

  const handleDeleteFileAndReset = async () => {
    try {
      const resp = await fetch(`${BASE_URL}/delete-excel`, {
        method: 'DELETE',
      });
      if (!resp.ok) throw new Error('Failed to delete file');
      setAllYearInputs([makeInitialBlock()]);
      setSelectedOption('startNew');
      setFileExists(false);
    } catch (err) {
      alert('Failed to delete existing file. Please try again.');
    }
  };

  const renderAssumptionBlock = (ent, idx) => {
    const isOpen = openBlocks[idx] ?? false;
    return (
      <Box
        key={ent.id}
        mb={4}
        p={2}
        border="1px solid #eee"
        borderRadius={2}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center">
            <IconButton
              size="small"
              onClick={() =>
                setOpenBlocks((ob) => ({
                  ...ob,
                  [idx]: !ob[idx],
                }))
              }
              sx={{
                transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s',
              }}
            >
              <ExpandMoreIcon />
            </IconButton>
            <TextField
              label="Year"
              type="number"
              size="small"
              sx={{ minWidth: 120, ml: 2 }}
              value={ent.year || ''}
              onChange={(e) => {
                const newYear = +e.target.value;
                setAllYearInputs((prev) => {
                  const copy = [...prev];
                  copy[idx].year = newYear;
                  return copy;
                });
              }}
              inputProps={{ min: 1900, max: 2100 }}
            />
          </Box>
          <IconButton
            size="small"
            onClick={() =>
              setAllYearInputs((prev) => prev.filter((_, blockIdx) => blockIdx !== idx))
            }
            disabled={allYearInputs.length === 1}
          >
            <DeleteIcon />
          </IconButton>
        </Box>

        <Collapse in={isOpen}>
          <TrafficRevenueCard
            year={ent.year}
            yearData={ent.traffic}
            onUpdateField={(field, value) => {
              const updated = { ...ent.traffic, [field]: parseFloat(value) || 0 };
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].traffic = updated;
                return copy;
              });
            }}
          />

          <MarketingExpensesCard
            yearData={ent.marketing}
            onUpdateField={(field, value) => {
              const updated = { ...ent.marketing, [field]: parseFloat(value) || 0 };
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].marketing = updated;
                return copy;
              });
            }}
          />

          <OfficeRentBreakdownCard
            data={ent.officeRentRows}
            onUpdate={(rowId, field, value) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                const newRows = copy[idx].officeRentRows.map((row) =>
                  row.id === rowId ? { ...row, [field]: parseFloat(value) || 0 } : row,
                );
                copy[idx].officeRentRows = newRows;
                return copy;
              });
            }}
            onAdd={() => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].officeRentRows = [
                  ...copy[idx].officeRentRows,
                  {
                    id: nextId.current++,
                    category: `Facility ${copy[idx].officeRentRows.length + 1}`,
                    squareMeters: 0,
                    costPerSQM: 0,
                  },
                ];
                return copy;
              });
            }}
            onRemove={(rowId) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].officeRentRows = copy[idx].officeRentRows.filter((row) => row.id !== rowId);
                return copy;
              });
            }}
          />

          <ProfessionalFeesCard
            data={ent.professionalFeesRows}
            onUpdate={(rowId, field, value) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                const newRows = copy[idx].professionalFeesRows.map((row) =>
                  row.id === rowId ? { ...row, [field]: field === 'Cost' ? parseFloat(value) || 0 : value } : row,
                );
                copy[idx].professionalFeesRows = newRows;
                return copy;
              });
            }}
            onAdd={() => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].professionalFeesRows = [
                  ...copy[idx].professionalFeesRows,
                  { id: nextId.current++, name: `Fee ${copy[idx].professionalFeesRows.length + 1}`, Cost: 0 },
                ];
                return copy;
              });
            }}
            onRemove={(rowId) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].professionalFeesRows = copy[idx].professionalFeesRows.filter((row) => row.id !== rowId);
                return copy;
              });
            }}
          />

          <DepreciationBreakdownCard
            data={ent.depreciationBreakdown}
            onUpdate={(assetId, field, value) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                const newDeps = copy[idx].depreciationBreakdown.map((asset) =>
                  asset.id === assetId ? { ...asset, [field]: parseFloat(value) || 0 } : asset,
                );
                copy[idx].depreciationBreakdown = newDeps;
                return copy;
              });
            }}
            onAdd={() => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].depreciationBreakdown = [
                  ...copy[idx].depreciationBreakdown,
                  {
                    id: nextId.current++,
                    name: `Asset ${copy[idx].depreciationBreakdown.length + 1}`,
                    amount: 0,
                    rate: 10,
                  },
                ];
                return copy;
              });
            }}
            onRemove={(assetId) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].depreciationBreakdown = copy[idx].depreciationBreakdown.filter((asset) => asset.id !== assetId);
                return copy;
              });
            }}
          />

          <BalanceSheetCard
            data={ent.balance}
            onUpdateField={(field, value) => {
              const updated = { ...ent.balance, [field]: parseFloat(value) || 0 };
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].balance = updated;
                return copy;
              });
            }}
          />

          <DebtIssuedCard
            data={ent.debtIssued}
            onUpdate={(debtId, field, value) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                const newDebt = copy[idx].debtIssued.map((debt) =>
                  debt.id === debtId ? { ...debt, [field]: parseFloat(value) || 0 } : debt,
                );
                copy[idx].debtIssued = newDebt;
                return copy;
              });
            }}
            onAdd={() => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].debtIssued = [
                  ...copy[idx].debtIssued,
                  {
                    id: nextId.current++,
                    name: `Debt ${copy[idx].debtIssued.length + 1}`,
                    amount: 0,
                    interestRate: 0,
                    duration: 1,
                  },
                ];
                return copy;
              });
            }}
            onRemove={(debtId) => {
              setAllYearInputs((prev) => {
                const copy = [...prev];
                copy[idx].debtIssued = copy[idx].debtIssued.filter((debt) => debt.id !== debtId);
                return copy;
              });
            }}
          />
        </Collapse>
      </Box>
    );
  };

  const renderInputPage = () => (
    <>
      {!fileExists && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Load an existing workbook or start a new plan from the sidebar to populate the financial model.
        </Alert>
      )}
      {allYearInputs.length === 0 ? (
        <Alert severity="warning">No input years defined. Add a year to begin configuring assumptions.</Alert>
      ) : (
        allYearInputs.map((ent, idx) => renderAssumptionBlock(ent, idx))
      )}

      <Box display="flex" gap={2} alignItems="center" sx={{ mt: 2 }}>
        <Button variant="outlined" onClick={addYearBlock}>
          Add Year
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={handleSaveAllData}
          disabled={saving || allYearInputs.length === 0}
        >
          {saving ? 'Saving…' : 'Save All Data'}
        </Button>
        {saveSuccess && (
          <Typography color="success.main">Saved!</Typography>
        )}
        {saveError && (
          <Typography color="error">{saveError}</Typography>
        )}
      </Box>
    </>
  );

  const renderKeyMetricsPage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Save or load assumptions to unlock scenario metrics and valuation summaries.
        </Alert>
      );
    }

    return (
      <>
        <SummaryCard
          kpisData={kpisData}
          summaryRowsData={summaryRowsData}
          loadingKpis={loadingKpis}
          loadingSummary={loadingSummary}
          errorKpis={errorKpis}
          errorSummary={errorSummary}
          scenario={scenario}
        />
        <MetricsAndExportCard />
      </>
    );
  };

  const renderFinancialPerformancePage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Run a base analysis first to view revenue, traffic, profitability, and waterfall trends.
        </Alert>
      );
    }

    return (
      <CombinedPerformanceCard
        loading={loadingCharts}
        error={chartsError}
        revenueData={allChartsData.revenue}
        trafficData={allChartsData.traffic}
        profitabilityData={allChartsData.profitability}
        breakEvenData={allChartsData.breakEven}
        considerationData={allChartsData.consideration}
        marginSafetyData={allChartsData.marginSafety}
        cashflowData={allChartsData.cashflow}
        profitMarginData={allChartsData.profitMargins}
        waterfallData={allChartsData.waterfall}
      />
    );
  };

  const renderFinancialPositionPage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Balance sheet views become available after assumptions are saved and analysed.
        </Alert>
      );
    }

    return (
      <>
        <FinancialSchedulesCard
          initialSelectedSchedules={['Balance Sheet']}
          title="Balance Sheet Overview"
          allowSelection={false}
        />
        <FinancialSchedulesCard
          initialSelectedSchedules={['Capital Assets', 'Debt Payment Schedule']}
          title="Capital Assets & Debt Schedules"
          allowSelection={false}
        />
      </>
    );
  };

  const renderCashFlowPage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Cash flow schedules are generated once the financial model has been run.
        </Alert>
      );
    }

    return (
      <FinancialSchedulesCard
        initialSelectedSchedules={['Cash Flow Statement']}
        title="Cash Flow Statement"
        allowSelection={false}
      />
    );
  };

  const renderSensitivityPage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Configure and save assumptions before exploring best and worst case sensitivities.
        </Alert>
      );
    }

    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <Typography variant="h6">Scenario Controls</Typography>
        <Typography variant="body2" color="text.secondary">
          Tune scenario multipliers and re-run the analysis from the sidebar to compare outcomes.
        </Typography>
        <BestCaseScenarioCard
          data={bestCaseData}
          onChange={(key, val) => setBestCaseData((prev) => ({ ...prev, [key]: val }))}
        />
        <WorstCaseScenarioCard
          data={worstCaseData}
          onChange={(key, val) => setWorstCaseData((prev) => ({ ...prev, [key]: val }))}
        />
      </Box>
    );
  };

  const renderAdvancedPage = () => {
    if (!fileExists) {
      return (
        <Alert severity="info">
          Load model data to unlock valuation, optimisation, Monte Carlo, and risk tooling.
        </Alert>
      );
    }

    return (
      <>
        <FinancialSchedulesCard
          initialSelectedSchedules={['Valuation', 'Customer Metrics']}
          title="Valuation & Customer Metrics"
        />
        <DCFValuationCard />
        <DetailedAnalysisCard
          discountRate={discountRate}
          wacc={wacc}
          growthRate={growthRate}
        />
        <AdvancedDecisionToolsCard
          discountRate={discountRate}
          wacc={wacc}
          perpetualGrowth={growthRate}
        />
      </>
    );
  };

  const renderActivePage = () => {
    switch (activePage) {
      case 'inputs':
        return renderInputPage();
      case 'metrics':
        return renderKeyMetricsPage();
      case 'performance':
        return renderFinancialPerformancePage();
      case 'position':
        return renderFinancialPositionPage();
      case 'cashflow':
        return renderCashFlowPage();
      case 'sensitivity':
        return renderSensitivityPage();
      case 'advanced':
        return renderAdvancedPage();
      default:
        return null;
    }
  };

  if (initialLoading) {
    return (
      <Box
        sx={{
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'background.default',
        }}
      >
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Loading your data, please wait…
        </Typography>
      </Box>
    );
  }

  return (
    <Box display="flex" height="100vh">
      <Sidebar
        selected={fileExists ? 'loadExisting' : 'startNew'}
        setSelected={setSelectedOption}
        hideAnalysisControls={!fileExists}
        onLoadExisting={handleLoadExisting}
        onStartNew={handleStartNew}
        onUploadFile={handleUploadFile}
        selectedFileName={null}
        discountRate={discountRate}
        setDiscountRate={setDiscountRate}
        wacc={wacc}
        setWacc={setWacc}
        growthRate={growthRate}
        setGrowthRate={setGrowthRate}
        taxRate={taxRate}
        setTaxRate={setTaxRate}
        inflationRate={inflationRate}
        setInflationRate={setInflationRate}
        laborRateIncrease={laborRateIncrease}
        setLaborRateIncrease={setLaborRateIncrease}
        scenario={scenario}
        setScenario={setScenario}
        analysisPeriod={analysisPeriod}
        setAnalysisPeriod={setAnalysisPeriod}
        startYear={startYear}
        setStartYear={setStartYear}
        onDeleteExistingFile={handleDeleteFileAndReset}
        setForecastYears={setForecastYears}
        forecastYears={forecastYears}
      />

      <Box flex={1} p={0} sx={{ height: '100vh', overflowY: 'auto' }}>
        <Card sx={{ mx: 4, mt: 3, mb: 2, borderRadius: 0 }}>
          <CardContent sx={{ px: 4, py: 3 }}>
            <Typography variant="h5" gutterBottom>
              Ecommerce Financial Analysis Dashboard
            </Typography>
            <Tabs
              value={activePage}
              onChange={(_, value) => setActivePage(value)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}
            >
              {PAGE_TABS.map((tab) => (
                <Tab key={tab.key} value={tab.key} label={tab.label} />
              ))}
            </Tabs>
            {renderActivePage()}
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
}
