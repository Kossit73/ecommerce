// src/components/AdvancedDecisionToolsCard.jsx

import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Grid,
  Button,
  TextField,
  MenuItem,
  Autocomplete,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import SidebarSlider from '../SidebarSlider';
import CustomNumericField from '../CustomNumericField';
import BASE_URL from '../../config';

const BUDGET_LINES = [
  'Total Marketing Budget',
  'COGS Budget',
  'Labor Budget',
  'Office Rent Budget',
];

const SENSITIVITY_OPTIONS = [
  'Average Item Value',
  'COGS Percentage',
  'Email Traffic',
  'Paid Search Traffic',
  'Email Conversion Rate',
  'Organic Search Conversion Rate',
  'Paid Search Conversion Rate',
  'Affiliates Conversion Rate',
];

export default function AdvancedDecisionToolsCard({
  discountRate,
  wacc,
  perpetualGrowth,
}) {
  const [open, setOpen] = useState(false);

  // ─── Sliders ───────────────────────────────────────────────────────────
  const [forecastYears, setForecastYears] = useState(10);
  const [sensitivityChange, setSensitivityChange] = useState(10);
  const [numSimulations, setNumSimulations] = useState(500);
  const [trafficIncrease, setTrafficIncrease] = useState(10);
  const [confidenceLevel, setConfidenceLevel] = useState(95);

  // ─── Dropdowns / multiselect / numeric ─────────────────────────────────
  const [budgetLine, setBudgetLine] = useState(BUDGET_LINES[0]);
  const [totalBudget, setTotalBudget] = useState(100000);
  const [sensitivityVars, setSensitivityVars] = useState([
    'Average Item Value',
    'COGS Percentage',
  ]);

  const adjustBudget = (delta) => {
    setTotalBudget((prev) => Math.max(0, prev + delta));
  };

  // ─── PrecisionTree state & function ────────────────────────────────────
  const [ptLoading, setPtLoading] = useState(false);
  const [decisionOutcomes, setDecisionOutcomes] = useState(null);
  const [decisionTreeImage, setDecisionTreeImage] = useState('');
  const [ptMessage, setPtMessage] = useState('');

  async function runPrecisionTree() {
    setPtLoading(true);
    setDecisionOutcomes(null);
    setDecisionTreeImage('');
    setPtMessage('');

    try {
      const resp = await fetch(
        `${BASE_URL}/precision_tree`,
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const {
        status,
        decision_outcomes,
        decision_tree_image,
        message,
      } = await resp.json();
      if (status !== 'success') {
        console.warn('PrecisionTree returned non-success status');
      }
      setDecisionOutcomes(decision_outcomes);
      setDecisionTreeImage(decision_tree_image);
      setPtMessage(message || '');
    } catch (err) {
      console.error('PrecisionTree failed:', err);
      setPtMessage(err.message);
    } finally {
      setPtLoading(false);
    }
  }

  // ─── NeuralTools state & function ───────────────────────────────────────
  const [ntLoading, setNtLoading] = useState(false);
  const [predictedRevenue, setPredictedRevenue] = useState(null);
  const [featureImportance, setFeatureImportance] = useState([]);
  const [ntMessage, setNtMessage] = useState('');

  async function runNeuralPrediction() {
    setNtLoading(true);
    setPredictedRevenue(null);
    setFeatureImportance([]);
    setNtMessage('');

    const payload = {
      traffic_increase_percentage: trafficIncrease,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/neural_tools`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const {
        status,
        predicted_revenue,
        feature_importance,
        message,
      } = await resp.json();
      if (status !== 'success') {
        console.warn('NeuralTools returned non-success status');
      }
      setPredictedRevenue(predicted_revenue);
      setFeatureImportance(feature_importance || []);
      setNtMessage(message || '');
    } catch (err) {
      console.error('NeuralTools prediction failed:', err);
      setNtMessage(err.message);
    } finally {
      setNtLoading(false);
    }
  }

  // ─── TopRank Sensitivity state & function ───────────────────────────────
  const [trLoading, setTrLoading] = useState(false);
  const [trResults, setTrResults] = useState([]);
  const [trInsights, setTrInsights] = useState([]);
  const [trMessage, setTrMessage] = useState('');

  async function runTopRankSensitivity() {
    setTrLoading(true);
    setTrResults([]);
    setTrInsights([]);
    setTrMessage('');

    if (sensitivityVars.length === 0) {
      alert('Please select at least one variable to test.');
      setTrLoading(false);
      return;
    }
    if (sensitivityChange <= 0) {
      alert('Change percentage must be > 0.');
      setTrLoading(false);
      return;
    }
    if (discountRate <= 0) {
      alert('Discount rate must be > 0.');
      setTrLoading(false);
      return;
    }

    const payload = {
      variables_to_test: sensitivityVars,
      change_percentage: sensitivityChange,
      discount_rate: discountRate,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/top_rank_sensitivity`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const { status, sensitivity_results, message, sensitivity_insights } =
        await resp.json();
      if (status !== 'success') {
        console.warn('TopRank returned non‐success status');
      }
      setTrResults(sensitivity_results);
      setTrInsights(sensitivity_insights || []);
      setTrMessage(message || '');
    } catch (err) {
      console.error('TopRank sensitivity failed:', err);
      setTrMessage(err.message);
    } finally {
      setTrLoading(false);
    }
  }

  // ─── Evolver Optimization state & function ──────────────────────────────
  const [evLoading, setEvLoading] = useState(false);
  const [evResults, setEvResults] = useState([]);
  const [evOriginalEbitda, setEvOriginalEbitda] = useState(null);
  const [evOptimizedEbitda, setEvOptimizedEbitda] = useState(null);
  const [evEbitdaChangePct, setEvEbitdaChangePct] = useState(null);
  const [evMessage, setEvMessage] = useState('');

  async function runEvolverOptimization() {
    setEvLoading(true);
    setEvResults([]);
    setEvOriginalEbitda(null);
    setEvOptimizedEbitda(null);
    setEvEbitdaChangePct(null);
    setEvMessage('');

    const payload = {
      budget_dict: { [budgetLine]: totalBudget },
      forecast_years: forecastYears,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/evolver_optimization`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const {
        status,
        optimized_data,
        original_ebitda,
        optimized_ebitda,
        ebitda_change_percent,
        message,
      } = await resp.json();
      if (status !== 'success') {
        console.warn('Evolver returned non-success status');
      }
      setEvResults(optimized_data || []);
      setEvOriginalEbitda(original_ebitda);
      setEvOptimizedEbitda(optimized_ebitda);
      setEvEbitdaChangePct(ebitda_change_percent);
      setEvMessage(message || '');
    } catch (err) {
      console.error('Evolver optimization failed:', err);
      setEvMessage(err.message);
    } finally {
      setEvLoading(false);
    }
  }

  // ─── StatTools Forecasting state & function ────────────────────────────
  const [stLoading, setStLoading] = useState(false);
  const [stForecastData, setStForecastData] = useState([]);
  const [stSummaryStats, setStSummaryStats] = useState({});
  const [stChartData, setStChartData] = useState([]);
  const [stMessage, setStMessage] = useState('');

  async function runStatToolsForecasting() {
    setStLoading(true);
    setStForecastData([]);
    setStSummaryStats({});
    setStChartData([]);
    setStMessage('');

    if (forecastYears < 1) {
      alert('Forecast years must be at least 1');
      setStLoading(false);
      return;
    }
    if (confidenceLevel < 80 || confidenceLevel > 99) {
      alert('Confidence level must be between 80 and 99');
      setStLoading(false);
      return;
    }

    const payload = {
      forecast_years: forecastYears,
      confidence_level: confidenceLevel,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/stat_tools_forecasting`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const {
        status,
        forecast_data,
        summary_statistics,
        format_dict,
        chart_data,
        message,
      } = await resp.json();
      setStForecastData(forecast_data);
      setStSummaryStats(summary_statistics || {});
      setStChartData(chart_data || []);
      setStMessage(message || '');
    } catch (err) {
      console.error('StatTools forecasting failed:', err);
      setStMessage(err.message);
    } finally {
      setStLoading(false);
    }
  }

  // ─── Schedule Risk state & function ────────────────────────────────────
  const [srLoading, setSrLoading] = useState(false);
  const [srSimulationDurations, setSrSimulationDurations] = useState([]);
  const [srMeanDuration, setSrMeanDuration] = useState(null);
  const [srCI, setSrCI] = useState({ lower: null, upper: null });
  const [srTasks, setSrTasks] = useState([]);
  const [srMessage, setSrMessage] = useState('');

  async function runScheduleRisk() {
    setSrLoading(true);
    setSrSimulationDurations([]);
    setSrMeanDuration(null);
    setSrCI({ lower: null, upper: null });
    setSrTasks([]);
    setSrMessage('');

    if (numSimulations < 100) {
      alert('Number of simulations must be at least 100');
      setSrLoading(false);
      return;
    }
    if (confidenceLevel < 80 || confidenceLevel > 99) {
      alert('Confidence level must be between 80 and 99');
      setSrLoading(false);
      return;
    }

    const payload = {
      num_simulations: numSimulations,
      confidence_level: confidenceLevel,
    };

    try {
      const resp = await fetch(
       `${BASE_URL}/schedule_risk_analysis`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server returned ${resp.status}: ${text}`);
      }
      const {
        status,
        simulation_durations,
        mean_duration,
        confidence_interval,
        tasks,
        message,
      } = await resp.json();
      setSrSimulationDurations(simulation_durations || []);
      setSrMeanDuration(mean_duration);
      setSrCI({ lower: confidence_interval.lower, upper: confidence_interval.upper });
      setSrTasks(tasks || []);
      setSrMessage(message || '');
    } catch (err) {
      console.error('ScheduleRisk failed:', err);
      setSrMessage(err.message);
    } finally {
      setSrLoading(false);
    }
  }

  // ───── Add refs to prevent initial-mount auto-run ────────────────────────
  const ntFirstRun = useRef(true);
  const trFirstRun = useRef(true);
  const evFirstRun = useRef(true);
  const stFirstRun = useRef(true);
  const srFirstRun = useRef(true);

  // ─── Auto-run NeuralPrediction when trafficIncrease changes ────────────
  useEffect(() => {
    if (ntFirstRun.current) {
      ntFirstRun.current = false;
      return;
    }
    runNeuralPrediction();
  }, [trafficIncrease]);

  // ─── Auto-run TopRank Sensitivity when sensitivityVars, sensitivityChange, or discountRate change ─
  useEffect(() => {
    if (trFirstRun.current) {
      trFirstRun.current = false;
      return;
    }
    runTopRankSensitivity();
  }, [JSON.stringify(sensitivityVars), sensitivityChange, discountRate]);

  // ─── Auto-run Evolver Optimization when budgetLine, totalBudget, or forecastYears change ────────
  useEffect(() => {
    if (evFirstRun.current) {
      evFirstRun.current = false;
      return;
    }
    runEvolverOptimization();
  }, [budgetLine, totalBudget, forecastYears]);

  // ─── Auto-run StatTools Forecasting when forecastYears or confidenceLevel change ───────────────
  useEffect(() => {
    if (stFirstRun.current) {
      stFirstRun.current = false;
      return;
    }
    runStatToolsForecasting();
  }, [forecastYears, confidenceLevel]);

  // ─── Auto-run Schedule Risk when numSimulations or confidenceLevel change ────────────────────
  useEffect(() => {
    if (srFirstRun.current) {
      srFirstRun.current = false;
      return;
    }
    runScheduleRisk();
  }, [numSimulations, confidenceLevel]);

  return (
    <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Advanced Decision Making Tools
        </Typography>

        <Accordion
          expanded={open}
          onChange={() => setOpen((o) => !o)}
          sx={{ mt: 2 }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Decision Making Tools Parameters</Typography>
          </AccordionSummary>

          <AccordionDetails>
            <Grid container spacing={12}>
              {/* Left column */}
              <Grid item xs={12} md={6}>
                <SidebarSlider
                  label="Forecast Years"
                  value={forecastYears}
                  setValue={setForecastYears}
                  min={1}
                  max={20}
                  step={1}
                />

                <SidebarSlider
                  label="Number of Simulations"
                  value={numSimulations}
                  setValue={setNumSimulations}
                  min={10}
                  max={10000}
                  step={100}
                />

                <SidebarSlider
                  label="Confidence Level (%)"
                  value={confidenceLevel}
                  setValue={setConfidenceLevel}
                  min={80}
                  max={99}
                  step={0.1}
                />

                <Box mt={1}>
                  <Typography variant="body2" fontWeight={500} gutterBottom>
                    Variables for Sensitivity Analysis
                  </Typography>
                  <Autocomplete
                    multiple
                    options={SENSITIVITY_OPTIONS}
                    value={sensitivityVars}
                    onChange={(e, v) => setSensitivityVars(v)}
                    renderInput={(params) => (
                      <TextField
                        {...params}
                        size="small"
                        placeholder="Select one or more…"
                        fullWidth
                      />
                    )}
                    sx={{ width: 340, height: 45 }}
                  />
                </Box>
              </Grid>

              {/* Right column */}
              <Grid item xs={12} md={6}>
                <SidebarSlider
                  label="Sensitivity Change Percentage"
                  value={sensitivityChange}
                  setValue={setSensitivityChange}
                  min={5}
                  max={20}
                  step={0.5}
                />

                <SidebarSlider
                  label="Traffic Increase for NeuralTools (%)"
                  value={trafficIncrease}
                  setValue={setTrafficIncrease}
                  min={5}
                  max={50}
                  step={1}
                />

                <Box mt={2}>
                  <Typography variant="body2" fontWeight={500} gutterBottom>
                    Select Budget Line for Optimization
                  </Typography>
                  <TextField
                    select
                    value={budgetLine}
                    onChange={(e) => setBudgetLine(e.target.value)}
                    size="small"
                    sx={{ width: 350 }}
                  >
                    {BUDGET_LINES.map((line) => (
                      <MenuItem key={line} value={line}>
                        {line}
                      </MenuItem>
                    ))}
                  </TextField>
                </Box>

                <Box display="flex" alignItems="center" mt={2}>
                  <CustomNumericField
                    label="Total Budget Amount ($)"
                    value={totalBudget}
                    onChange={(val) => setTotalBudget(val)}
                    step={1000}
                    width={350}
                    height={30}
                  />
                </Box>
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>

        {/* ─── Action buttons ────────────────────────────────────────────────── */}
        <Grid container spacing={2} mt={3}>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runPrecisionTree}
              disabled={ptLoading}
            >
              {ptLoading ? 'Running Tree Analysis...' : 'Run Tree Analysis'}
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runNeuralPrediction}
              disabled={ntLoading}
            >
              {ntLoading
                ? 'Running Neural Prediction...'
                : 'Run Neural Prediction'}
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runTopRankSensitivity}
              disabled={trLoading}
            >
              {trLoading
                ? 'Running TopRank...'
                : 'Run TopRank Sensitivity Analysis'}
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runEvolverOptimization}
              disabled={evLoading}
            >
              {evLoading ? 'Running Optimization…' : 'Run Optimization'}
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runStatToolsForecasting}
              disabled={stLoading}
            >
              {stLoading ? 'Running Forecast…' : 'Run Forecasting Stats'}
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={runScheduleRisk}
              disabled={srLoading}
            >
              {srLoading
                ? 'Running Schedule Risk…'
                : 'Run Risk Analysis (Schedule Risk)'}
            </Button>
          </Grid>
        </Grid>

        {/* ─── PrecisionTree results ────────────────────────────────────────── */}
        {decisionOutcomes && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Decision Outcomes
            </Typography>
            <Box component="ul" sx={{ pl: 2 }}>
              {Object.entries(decisionOutcomes).map(([label, value]) => (
                <Typography key={label} component="li">
                  {label}: ${value.toFixed(2)}
                </Typography>
              ))}
            </Box>
            {decisionTreeImage && (
              <Box mt={2} textAlign="center">
                <img
                  src={decisionTreeImage}
                  alt="PrecisionTree"
                  style={{ maxWidth: '100%', height: 'auto' }}
                />
              </Box>
            )}
            {ptMessage && (
              <Box mt={2}>
                <Typography color="primary">{ptMessage}</Typography>
              </Box>
            )}
          </Box>
        )}

        {/* ─── NeuralTools results ──────────────────────────────────────────── */}
        {predictedRevenue != null && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Neural Prediction Results
            </Typography>
            {ntMessage && (
              <Typography color="primary" gutterBottom>
                {ntMessage}
              </Typography>
            )}
            <Typography gutterBottom>
              Predicted Revenue: ${predictedRevenue.toFixed(2)}
            </Typography>
            {featureImportance.length > 0 && (
              <Box mt={2}>
                <Typography variant="subtitle1" gutterBottom>
                  Feature Importance
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Feature</TableCell>
                      <TableCell>Importance</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {featureImportance.map((item, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{item.feature}</TableCell>
                        <TableCell>
                          {(item.importance * 100).toFixed(2)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Box>
            )}
          </Box>
        )}

        {/* ─── TopRank Sensitivity results ──────────────────────────────────── */}
        {trResults.length > 0 && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              TopRank Sensitivity Results
            </Typography>
            {trMessage && (
              <Box mb={2}>
                <Typography color="primary">{trMessage}</Typography>
              </Box>
            )}
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Variable</TableCell>
                  <TableCell>Direction</TableCell>
                  <TableCell>Net Income Δ (%)</TableCell>
                  <TableCell>EBITDA Δ (%)</TableCell>
                  <TableCell>Cash Flow Δ (%)</TableCell>
                  <TableCell>Equity Value Δ (%)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {trResults.map((row, i) => (
                  <TableRow key={i}>
                    <TableCell>{row.variable}</TableCell>
                    <TableCell>{row.direction}</TableCell>
                    <TableCell>{row.net_income_change.toFixed(2)}</TableCell>
                    <TableCell>{row.ebitda_change.toFixed(2)}</TableCell>
                    <TableCell>{row.net_cash_flow_change.toFixed(2)}</TableCell>
                    <TableCell>{row.equity_value_change.toFixed(2)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {trInsights.length > 0 && (
              <Box mt={2}>
                <Typography variant="subtitle1">Insights:</Typography>
                <Box component="ul" sx={{ pl: 2 }}>
                  {trInsights.map((line, idx) => (
                    <Typography key={idx} component="li">
                      {line}
                    </Typography>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        )}

        {/* ─── Evolver Optimization results ──────────────────────────────────── */}
        {evResults.length > 0 && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Evolver Optimization Results
            </Typography>
            {evMessage && (
              <Box mb={2}>
                <Typography color="primary">{evMessage}</Typography>
              </Box>
            )}
            <Typography gutterBottom>
              Original EBITDA: ${evOriginalEbitda.toFixed(2)}
            </Typography>
            <Typography gutterBottom>
              Optimized EBITDA: ${evOptimizedEbitda.toFixed(2)}
            </Typography>
            <Typography gutterBottom>
              EBITDA Δ: {evEbitdaChangePct.toFixed(2)}%
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Year</TableCell>
                  <TableCell>Variables (Traffic)</TableCell>
                  <TableCell>Change (%)</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {evResults.map((row, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{row.year}</TableCell>
                    <TableCell>
                      {Object.entries(row.variables).map(([varName, val]) => (
                        <div key={varName}>
                          {varName}: {val.toFixed(0)}
                        </div>
                      ))}
                    </TableCell>
                    <TableCell>
                      {Object.entries(row.changes).map(([varName, pct]) => (
                        <div key={varName}>
                          {varName}: {pct.toFixed(2)}%
                        </div>
                      ))}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}

        {/* ─── StatTools Forecasting results ──────────────────────────────────── */}
        {stForecastData.length > 0 && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              StatTools Forecasting Results
            </Typography>
            {stMessage && (
              <Box mb={2}>
                <Typography color="primary">{stMessage}</Typography>
              </Box>
            )}

            <Typography variant="subtitle1" gutterBottom>
              Summary Statistics
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell>Mean Forecast</TableCell>
                  <TableCell>Last Forecast</TableCell>
                  <TableCell>CI Width</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(stSummaryStats).map(([metric, stats]) => {
                  const ciKey = Object.keys(stats).find((k) =>
                    k.endsWith('CI Width')
                  );
                  return (
                    <TableRow key={metric}>
                      <TableCell>{metric}</TableCell>
                      <TableCell>
                        {typeof stats['Mean Forecast'] === 'number'
                          ? stats['Mean Forecast'].toFixed(0)
                          : '-'}
                      </TableCell>
                      <TableCell>
                        {typeof stats['Last Forecast'] === 'number'
                          ? stats['Last Forecast'].toFixed(0)
                          : '-'}
                      </TableCell>
                      <TableCell>
                        {ciKey && typeof stats[ciKey] === 'number'
                          ? stats[ciKey].toFixed(0)
                          : '-'}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>

            <Typography variant="body2" mt={2}>
              (Check DevTools console for detailed `chart_data`.)
            </Typography>
          </Box>
        )}

        {/* ─── Schedule Risk results ──────────────────────────────────────────── */}
        {srMeanDuration != null && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Schedule Risk Analysis Results
            </Typography>
            {srMessage && (
              <Box mb={2}>
                <Typography color="primary">{srMessage}</Typography>
              </Box>
            )}
            <Typography gutterBottom>
              Mean Completion Time: {srMeanDuration.toFixed(1)} days
            </Typography>
            <Typography gutterBottom>
              {confidenceLevel}% CI: [ {srCI.lower.toFixed(1)} ,{' '}
              {srCI.upper.toFixed(1)} ] days
            </Typography>

            <Typography variant="subtitle1" gutterBottom mt={2}>
              Task Definitions
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Task Name</TableCell>
                  <TableCell>Base Duration</TableCell>
                  <TableCell>Std Dev</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {srTasks.map((t, idx) => (
                  <TableRow key={idx}>
                    <TableCell>{t.name}</TableCell>
                    <TableCell>{t.base_duration.toFixed(1)}</TableCell>
                    <TableCell>{t.std_dev.toFixed(1)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}

        {/* ─── Export Options ────────────────────────────────────────────────── */}
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Export Options 🔗
          </Typography>
          <Button variant="outlined" color="primary">
            Download Detailed Analysis Report
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
}
