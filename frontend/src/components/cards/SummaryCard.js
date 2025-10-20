// src/components/SummaryCard.jsx

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  CircularProgress,
} from '@mui/material';

// Utility to format millions: 2.345 → “$2.3M”
function formatMillions(val) {
  if (val == null || isNaN(val)) return '—';
  return `$${val.toFixed(1)}M`;
}

// Utility to format percentages: 12.345 → “12.35%”
function formatPercent(val) {
  if (val == null || isNaN(val)) return '—';
  return `${val.toFixed(2)}%`;
}

export default function SummaryCard({
  kpisData,
  summaryRowsData,
  loadingKpis,
  loadingSummary,
  errorKpis,
  errorSummary,
  scenario
}) {
  // 1) If either part is still loading → show spinner
  if (loadingKpis || loadingSummary) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box textAlign="center" py={4}>
            <CircularProgress />
            <Typography mt={1}>Loading summary…</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // 2) If any error occurred → show error(s)
  if (errorKpis || errorSummary) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          {errorKpis && (
            <Typography color="error" gutterBottom>
              KPI Error: {errorKpis}
            </Typography>
          )}
          {errorSummary && (
            <Typography color="error">
              Summary Table Error: {errorSummary}
            </Typography>
          )}
        </CardContent>
      </Card>
    );
  }

  // 3) If data has not yet arrived (both are still null) → show spinner as well
  if (!kpisData || !summaryRowsData) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box textAlign="center" py={4}>
            <CircularProgress />
            <Typography mt={1}>Loading summary…</Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  // 4) Both kpisData AND summaryRowsData are present → render the actual content
  const {
    revenue_growth,
    current_revenue_growth,
    ebitda_margin,
    ebitda_margin_delta,
    enterprise_value,
    estimated_enterprise_value,
  } = kpisData;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Scenario Analysis: {scenario}
        </Typography>

        {/* Top‐level KPI cards */}
        <Box display="flex" justifyContent="space-between" my={2}>
          <Box>
            <Typography variant="h6">Revenue Growth</Typography>
            <Typography
              variant="h4"  
              color={revenue_growth >= 0 ? 'success.main' : 'error.main'}
            >
              {formatPercent(revenue_growth)}
            </Typography>
            <Typography
              color={current_revenue_growth  >= 0  ? 'success.main' : 'error.main'}
            >
              {current_revenue_growth >= 0 ? '↑ ' : '↓ '}
              {formatPercent(Math.abs(current_revenue_growth))}
            </Typography>
          </Box>

          <Box>
            <Typography variant="h6">EBITDA Margin</Typography>
            <Typography variant="h4">{formatPercent(ebitda_margin)}</Typography>
            <Typography
              color={ebitda_margin_delta >= 0 ? 'success.main' : 'error.main'}
            >
              {ebitda_margin_delta >= 0 ? '↑ ' : '↓ '}
              {formatPercent(Math.abs(ebitda_margin_delta))}
            </Typography>
          </Box>

          <Box>
            <Typography variant="h6">Enterprise Value</Typography>
            <Typography variant="h4">
              {formatMillions(
                enterprise_value != null
                  ? enterprise_value
                  : estimated_enterprise_value
              )}
            </Typography>
            {estimated_enterprise_value != null && (
              <Typography color="text.secondary" variant="body2">
                (est.)
              </Typography>
            )}
          </Box>
        </Box>

        {/* “Summary of Analyses” Table */}
        <Typography variant="h6" mt={4}>
          Summary of Analyses
        </Typography>
        <Table size="small" sx={{ mt: 2 }}>
          <TableHead>
            <TableRow>
              <TableCell>Scenario</TableCell>
              <TableCell>Net Income ($M)</TableCell>
              <TableCell>EBITDA ($M)</TableCell>
              <TableCell>IRR (%)</TableCell>
              <TableCell>NPV ($M)</TableCell>
              <TableCell>Payback Period (Orders)</TableCell>
              <TableCell>Gross Profit Margin (%)</TableCell>
              <TableCell>Net Profit Margin (%)</TableCell>
              <TableCell>Net Cash Flow ($M)</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {summaryRowsData.map((row, i) => (
              <TableRow key={i}>
                <TableCell>{row.Scenario}</TableCell>
                <TableCell>{formatMillions(row.Net_Income_M)}</TableCell>
                <TableCell>{formatMillions(row.EBITDA_M)}</TableCell>
                <TableCell>{formatPercent(row.IRR)}</TableCell>
                <TableCell>{formatMillions(row.NPV_M)}</TableCell>
                <TableCell>
                  {row.Payback_Period_Orders?.toFixed(2) ?? '—'}
                </TableCell>
                <TableCell>{formatPercent(row.Gross_Profit_Margin)}</TableCell>
                <TableCell>{formatPercent(row.Net_Profit_Margin)}</TableCell>
                <TableCell>{formatMillions(row.Net_Cash_Flow_M)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        {/* Key Implications */}
        <Typography variant="h6" mt={4}>
          Key Implications
        </Typography>
        <ul>
          <li>
            <strong>Base Case:</strong> Stable performance baseline with moderate
            growth and risk.
          </li>
          <li>
            <strong>Best Case:</strong> Optimistic growth with improved
            efficiencies and reduced costs.
          </li>
          <li>
            <strong>Worst Case:</strong> Challenging conditions with higher costs
            and reduced revenue.
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}
