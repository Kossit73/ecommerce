// src/components/DetailedAnalysisCard.jsx

import React, { useState, useEffect, useRef  } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Button,
  IconButton,
  TextField,
  Slider,
  Select,
  MenuItem,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';
import SidebarSlider from '../SidebarSlider';
import BASE_URL from '../../config';
export default function DetailedAnalysisCard({
  discountRate, // passed from parent
  wacc,         // passed from parent
  growthRate,   // passed from parent (perpetualGrowth)
}) {
  const VARIABLE_OPTIONS = [
    'Total Orders',
  'Average Item Value',
  'Email Conversion Rate',
  'Paid Search Traffic',
  'COGS Percentage',
  'Labor/Handling per Order',
  'Freight/Shipping per Order',
  'Marketing Expenses',
  'Interest Rate',
  ];
  const DISTRIBUTIONS = [ "Normal", "Lognormal", "Uniform", "Exponential",
  "Binomial", "Poisson", "Geometric", "Bernoulli",
  "Chi-square", "Gamma", "Weibull", "Hypergeometric",
  "Multinomial", "Beta", "F-distribution", "Discrete",
  "Continuous", "Cumulative"];

  // ─── Analysis Parameters Accordion ─────────────────────────────────────
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [forecastYears, setForecastYears] = useState(10);
  const [numSimulations, setNumSimulations] = useState(500);
  const [confidenceLevel, setConfidenceLevel] = useState(95);
  const [distribution, setDistribution] = useState('Normal');

  // ─── What-If Analysis State ────────────────────────────────────────────
  const [numAdjustments, setNumAdjustments] = useState(1);
  const [adjustments, setAdjustments] = useState([
    { year: 2025, variable: VARIABLE_OPTIONS[0], multiplier: 1 },
  ]);

  const updateAdjustment = (idx, key, value) => {
    setAdjustments((adjs) => {
      const copy = [...adjs];
      copy[idx] = { ...copy[idx], [key]: value };
      return copy;
    });
  };

  const changeNum = (delta) => {
    setNumAdjustments((n) => {
      const next = Math.max(1, n + delta);
      setAdjustments((adjs) => {
        let c = [...adjs];
        while (c.length < next) {
          c.push({ year: 2025, variable: VARIABLE_OPTIONS[0], multiplier: 1 });
        }
        return c.slice(0, next);
      });
      return next;
    });
  };

  // ─── Goal Seek State ───────────────────────────────────────────────────
  const [goalOpen, setGoalOpen] = useState(false);
  const [goalYear, setGoalYear] = useState(2025);
  const [targetMargin, setTargetMargin] = useState(15);
  const [seekVariable, setSeekVariable] = useState(VARIABLE_OPTIONS[0]);

  const [gsLoading, setGsLoading] = useState(false);
  const [gsResults, setGsResults] = useState([]);
  const [gsCurrentMargin, setGsCurrentMargin] = useState(null);
  const [gsTargetMarginAchieved, setGsTargetMarginAchieved] = useState(null);
  const [gsMultiplier, setGsMultiplier] = useState(null);
  const [gsMessage, setGsMessage] = useState('');
  const [gsWarnings, setGsWarnings] = useState([]);

  // ─── Monte Carlo State ──────────────────────────────────────────────────
  const [mcLoading, setMcLoading] = useState(false);
  const [mcMetrics, setMcMetrics] = useState(null);
  const [mcFinalIncome, setMcFinalIncome] = useState([]);
  const [mcNpvValues, setMcNpvValues] = useState([]);
  const [mcMeanIncome, setMcMeanIncome] = useState(null);
  const [mcCiValues, setMcCiValues] = useState([]);
  const [mcMeanNpv, setMcMeanNpv] = useState(null);
  const [mcCiNpv, setMcCiNpv] = useState([]);

  // ─── What-If State ─────────────────────────────────────────────────────
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfResults, setWhatIfResults] = useState([]);
  const [whatIfWarnings, setWhatIfWarnings] = useState([]);
 const mcFirstRun = useRef(true);
  const whatIfFirstRun = useRef(true);
  const gsFirstRun = useRef(true);

  // ──────────────────────────────────────────────────────────────────────
  // 1) Monte Carlo API call
  // ──────────────────────────────────────────────────────────────────────
  async function runMonteCarlo() {
    setMcLoading(true);

    const payload = {
      forecast_years: forecastYears,
      num_simulations: numSimulations,
      confidence_level: confidenceLevel,
      distribution_type: distribution,
      discount_rate: discountRate/100,
      wacc: wacc/100,
      perpetual_growth: growthRate/100,
    };

    try {
       console.log("payload", payload)
      const resp = await fetch(
        
        
        `${BASE_URL}/monte_carlo`,
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
        metrics,
        final_year_income,
        npv_values,
        mean_income,
        ci_values,
        mean_npv,
        ci_npv,
      } = await resp.json();

      if (status !== 'success') {
        console.warn('Monte Carlo returned non-success status');
      }

      setMcMetrics(metrics);
      setMcFinalIncome(final_year_income);
      setMcNpvValues(npv_values);
      setMcMeanIncome(mean_income);
      setMcCiValues(ci_values);
      setMcMeanNpv(mean_npv);
      setMcCiNpv(ci_npv);
    } catch (err) {
      console.error('Monte Carlo failed:', err);
    } finally {
      setMcLoading(false);
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  // 2) What-If API call
  // ──────────────────────────────────────────────────────────────────────
  async function runWhatIf() {
    setWhatIfLoading(true);

    const payload = {
      num_adjustments: numAdjustments,
      adjustments: adjustments.map((adj) => ({
        year: adj.year,
        variable: adj.variable,
        multiplier: adj.multiplier,
      })),
      discount_rate: discountRate/100,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/what_if`,
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

      const { status, results, warnings } = await resp.json();
      if (status !== 'success') {
        console.warn('What‐If returned non-success status');
      }

      setWhatIfResults(results || []);
      setWhatIfWarnings(warnings || []);
    } catch (err) {
      console.error('What‐If failed:', err);
    } finally {
      setWhatIfLoading(false);
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  // 3) Goal Seek API call
  // ──────────────────────────────────────────────────────────────────────
  async function runGoalSeek() {
    setGsLoading(true);
    setGsResults([]);
    setGsCurrentMargin(null);
    setGsTargetMarginAchieved(null);
    setGsMultiplier(null);
    setGsMessage('');
    setGsWarnings([]);

    const payload = {
      target_profit_margin: targetMargin,
      variable_to_adjust: seekVariable,
      year_to_adjust: goalYear,
      max_iterations: 100,
      tolerance: 0.001,
      discount_rate: discountRate/100,
    };

    try {
      const resp = await fetch(
        `${BASE_URL}/goal_seek`,
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
        current_profit_margin,
        target_profit_margin,
        multiplier,
        results,
        message,
        warnings,
      } = await resp.json();

      if (status !== 'success') {
        console.warn('Goal Seek returned non-success status:', message);
      }

      setGsCurrentMargin(current_profit_margin);
      setGsTargetMarginAchieved(target_profit_margin);
      setGsMultiplier(multiplier);
      setGsResults(results || []);
      setGsMessage(message || '');
      setGsWarnings(warnings || []);
    } catch (err) {
      console.error('Goal Seek failed:', err);
      setGsMessage(err.message);
    } finally {
      setGsLoading(false);
    }
  }

  // ──────────────────────────────────────────────────────────────────────
  // 4) Auto‐run Monte Carlo whenever its inputs change
  // ──────────────────────────────────────────────────────────────────────
  useEffect(() => {
  if (mcFirstRun.current) {
    // Skip the very first invocation
    mcFirstRun.current = false;
    return;
  }
  runMonteCarlo();
}, [
  forecastYears,
  numSimulations,
  confidenceLevel,
  distribution,
  discountRate,
  wacc,
  growthRate,
]);

  // ──────────────────────────────────────────────────────────────────────
  // 5) Auto‐run What‐If whenever `adjustments` or `discountRate` change
  // ──────────────────────────────────────────────────────────────────────
  // useEffect(() => {
  //   runWhatIf();
  // }, [
  //   // Stringify so any deep change in adjustments array triggers effect
  //   JSON.stringify(adjustments),
  //   discountRate,
  // ]);
 useEffect(() => {
  if (whatIfFirstRun.current) {
    whatIfFirstRun.current = false;
    return;
  }
  runWhatIf();
}, [JSON.stringify(adjustments), discountRate]);
  // ──────────────────────────────────────────────────────────────────────
  // 6) Auto‐run Goal Seek whenever its inputs change
  // ──────────────────────────────────────────────────────────────────────
 
useEffect(() => {
  if (gsFirstRun.current) {
    // first time, just flip the flag and bail out
    gsFirstRun.current = false;
    return;
  }
  // subsequent changes will trigger this
  runGoalSeek();
}, [goalYear, targetMargin, seekVariable, discountRate]);
  // ──────────────────────────────────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────────────────────────────────
  return (
    <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Detailed Analysis
        </Typography>

        {/* ─── Analysis Parameters ───────────────────────────────────────────── */}
        <Accordion
          expanded={analysisOpen}
          onChange={() => setAnalysisOpen((o) => !o)}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography>Analysis Parameters</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ maxWidth: 600, width: '100%', pr: 2 }}>
              {/* Forecast Years */}
              <SidebarSlider
                label="Forecast Years"
                value={forecastYears}
                setValue={setForecastYears}
                min={1}
                max={20}
                step={1}
              />

              {/* Number of Simulations */}
              <SidebarSlider
                label="Number of Simulations"
                value={numSimulations}
                setValue={setNumSimulations}
                min={100}
                max={5000}
                step={100}
              />

              {/* Confidence Level */}
              <SidebarSlider
                label="Confidence Level (%)"
                value={confidenceLevel}
                setValue={setConfidenceLevel}
                min={80}
                max={99}
                step={0.1}
              />

              {/* Distribution Select */}
              <Box mt={2}>
                <Typography variant="body2" gutterBottom>
                  Select Distribution for Monte Carlo Simulation
                </Typography>
                <Select
                  value={distribution}
                  onChange={(e) => setDistribution(e.target.value)}
                  fullWidth
                  size="small"
                >
                  {DISTRIBUTIONS.map((d) => (
                    <MenuItem key={d} value={d}>
                      {d}
                    </MenuItem>
                  ))}
                </Select>
              </Box>
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* ─── Manual Monte Carlo Trigger (still available) ───────────────── */}
        <Box mt={2}>
          <Button
            variant="outlined"
            color="primary"
            onClick={runMonteCarlo}
            disabled={mcLoading}
          >
            {mcLoading
              ? 'Running Simulation...'
              : 'Run Monte Carlo Simulation'}
          </Button>
        </Box>

        {/* ─── Monte Carlo Results ──────────────────────────────────────────── */}
        {mcMetrics && (
          <Box mt={3}>
            <Typography variant="h6" gutterBottom>
              Monte Carlo Results (Final‐Year Metrics)
            </Typography>
            <Typography>
              Mean Final‐Year Income: {mcMeanIncome?.toFixed(2)}{' '}
              (CI: {mcCiValues[0].toFixed(2)} – {mcCiValues[1].toFixed(2)})
            </Typography>
            <Typography>
              Mean NPV: {mcMeanNpv?.toFixed(2)}{' '}
              (CI: {mcCiNpv[0].toFixed(2)} – {mcCiNpv[1].toFixed(2)})
            </Typography>

            <Box mt={2}>
              <Typography variant="subtitle1">Full Metric Summaries:</Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell>Mean</TableCell>
                    <TableCell>CI Lower</TableCell>
                    <TableCell>CI Upper</TableCell>
                    <TableCell>Unit</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(mcMetrics).map(([metricName, summary]) => (
                    <TableRow key={metricName}>
                      <TableCell>{metricName}</TableCell>
                      <TableCell>{summary.mean.toFixed(2)}</TableCell>
                      <TableCell>{summary.ci_lower.toFixed(2)}</TableCell>
                      <TableCell>{summary.ci_upper.toFixed(2)}</TableCell>
                      <TableCell>{summary.unit}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          </Box>
        )}

        {/* ─── What-If Analysis ──────────────────────────────────────────────── */}
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            What‐If Analysis for E‐commerce
          </Typography>

          <Box display="flex" alignItems="center" gap={1} mb={2}>
            <Typography>Number of Adjustments</Typography>
            <IconButton size="small" onClick={() => changeNum(-1)}>
              <RemoveIcon />
            </IconButton>
            <Typography>{numAdjustments}</Typography>
            <IconButton size="small" onClick={() => changeNum(1)}>
              <AddIcon />
            </IconButton>
          </Box>

          {adjustments.map((adj, i) => (
            <Accordion key={i} defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Adjustment {i + 1}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box display="flex" flexWrap="wrap" gap={2}>
                  <TextField
                    label="Year"
                    type="number"
                    size="small"
                    value={adj.year}
                    onChange={(e) =>
                      updateAdjustment(i, 'year', +e.target.value)
                    }
                  />
                  <Select
                    label="Variable"
                    size="small"
                    value={adj.variable}
                    onChange={(e) =>
                      updateAdjustment(i, 'variable', e.target.value)
                    }
                  >
                    {VARIABLE_OPTIONS.map((v) => (
                      <MenuItem key={v} value={v}>
                        {v}
                      </MenuItem>
                    ))}
                  </Select>
                  <Box flexGrow={1}>
                    <Typography gutterBottom>
                      Multiplier: {adj.multiplier.toFixed(2)}
                    </Typography>
                    <Slider
                      min={0.5}
                      max={1.5}
                      step={0.01}
                      value={adj.multiplier}
                      onChange={(e, v) =>
                        updateAdjustment(i, 'multiplier', v)
                      }
                    />
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}

          <Box mt={2}>
            <Button
              variant="outlined"
              color="primary"
              onClick={runWhatIf}
              disabled={whatIfLoading}
            >
              {whatIfLoading
                ? 'Running What‐If...'
                : 'Run What‐If Analysis'}
            </Button>
          </Box>

          {whatIfResults.length > 0 && (
            <Box mt={3}>
              <Typography variant="subtitle1" gutterBottom>
                What‐If Analysis Results:
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Year</TableCell>
                    <TableCell>Net Revenue</TableCell>
                    <TableCell>Gross Profit</TableCell>
                    <TableCell>EBITDA</TableCell>
                    <TableCell>Net Income</TableCell>
                    <TableCell>Total Orders</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {whatIfResults.map((row, i) => (
                    <TableRow key={i}>
                      <TableCell>{row.year}</TableCell>
                      <TableCell>{row.net_revenue.toFixed(2)}</TableCell>
                      <TableCell>{row.gross_profit.toFixed(2)}</TableCell>
                      <TableCell>{row.ebitda.toFixed(2)}</TableCell>
                      <TableCell>{row.net_income.toFixed(2)}</TableCell>
                      <TableCell>{row.total_orders.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {whatIfWarnings.length > 0 && (
                <Box mt={2}>
                  {whatIfWarnings.map((w, idx) => (
                    <Typography key={idx} color="warning.main">
                      ⚠️ {w}
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* ─── Goal Seek Analysis ─────────────────────────────────────────────── */}
        <Box mt={4}>
          <Typography variant="h6" gutterBottom>
            Goal Seek Analysis
          </Typography>

          <Accordion
            expanded={goalOpen}
            onChange={() => setGoalOpen((o) => !o)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography>Goal Seek Parameters</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box display="flex" flexWrap="wrap" gap={2}>
                <TextField
                  label="Year to Adjust"
                  type="number"
                  size="small"
                  value={goalYear}
                  onChange={(e) => setGoalYear(+e.target.value)}
                />
                <Box flexGrow={1}>
                  <Typography gutterBottom>
                    Target Profit Margin Increase (%)
                  </Typography>
                  <Slider
                    min={0}
                    max={50}
                    step={0.1}
                    value={targetMargin}
                    onChange={(e, v) => setTargetMargin(v)}
                  />
                </Box>
                <Select
                  label="Variable to Adjust"
                  size="small"
                  value={seekVariable}
                  onChange={(e) => setSeekVariable(e.target.value)}
                >
                  {VARIABLE_OPTIONS.map((v) => (
                    <MenuItem key={v} value={v}>
                      {v}
                    </MenuItem>
                  ))}
                </Select>
              </Box>
            </AccordionDetails>
          </Accordion>
              <Box mt={2}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={runGoalSeek}
                  disabled={gsLoading}
                >
                  {gsLoading ? 'Running Goal Seek...' : 'Run Goal Seek'}
                </Button>
              </Box>

              {gsMessage && (
                <Box mt={2}>
                  <Typography color="primary">{gsMessage}</Typography>
                </Box>
              )}

              {gsCurrentMargin != null && (
                <Box mt={2}>
                  <Typography>
                    Current Profit Margin for {goalYear}:{' '}
                    {(gsCurrentMargin * 100).toFixed(2)}%
                  </Typography>
                  <Typography>
                    Target Profit Margin:{' '}
                    {(gsTargetMarginAchieved * 100).toFixed(2)}%
                  </Typography>
                </Box>
              )}

              {gsResults.length > 0 && (
                <Box mt={3}>
                  <Typography variant="subtitle1" gutterBottom>
                    Goal Seek Results:
                  </Typography>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Year</TableCell>
                        <TableCell>Net Revenue</TableCell>
                        <TableCell>Gross Profit</TableCell>
                        <TableCell>EBITDA</TableCell>
                        <TableCell>Net Income</TableCell>
                        <TableCell>Total Orders</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {gsResults.map((row, i) => (
                        <TableRow key={i}>
                          <TableCell>{row.year}</TableCell>
                          <TableCell>{row.net_revenue.toFixed(2)}</TableCell>
                          <TableCell>{row.gross_profit.toFixed(2)}</TableCell>
                          <TableCell>{row.ebitda.toFixed(2)}</TableCell>
                          <TableCell>{row.net_income.toFixed(2)}</TableCell>
                          <TableCell>{row.total_orders.toFixed(2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>

                  {gsWarnings.length > 0 && (
                    <Box mt={2}>
                      {gsWarnings.map((w, idx) => (
                        <Typography key={idx} color="warning.main">
                          ⚠️ {w}
                        </Typography>
                      ))}
                    </Box>
                  )}
                </Box>
              )}
           
        </Box>

        {/* ─── Export Options ───────────────────────────────────────────────── */}
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
