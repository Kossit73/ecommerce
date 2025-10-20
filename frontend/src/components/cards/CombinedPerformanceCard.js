// src/components/CombinedPerformanceCard.jsx

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
} from '@mui/material';
import Plot from 'react-plotly.js';

export default function CombinedPerformanceCard({
  loading,
  error,
  revenueData,
  trafficData,
  profitabilityData,
  breakEvenData,
  considerationData,
  marginSafetyData,
  cashflowData,
  profitMarginData,
  waterfallData,
}) {
  const plotConfig = { displaylogo: false, responsive: true };
  const baseLayout = {
    margin: { l: 40, r: 40, t: 30, b: 80 },
    hovermode: 'x unified',
  };

  if (loading) {
    return (
      <Box textAlign="center" my={2}>
        <CircularProgress />
        <Typography mt={1}>Loading charts…</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Typography color="error" textAlign="center" sx={{ my: 2 }}>
        Error loading charts: {error}
      </Typography>
    );
  }

  return (
    <div id="combined-performance-chart">
    <Card sx={{ mx: 1, my: 5, borderRadius: 0 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Financial Performance Analysis
        </Typography>

        <Box
          display="grid"
          gridTemplateColumns="repeat(3, 1fr)"
          gap={3}
          mt={2}
        >
          {revenueData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Revenue & Profitability
              </Typography>
              <Plot
                data={[
                  {
                    x: revenueData.years,
                    y: revenueData.net_revenue,
                    type: 'bar',
                    name: 'Net Revenue',
                    marker: { color: '#8884d8' },
                    yaxis: 'y1',
                  },
                  {
                    x: revenueData.years,
                    y: revenueData.gross_margin.map((m) => m / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Gross Margin',
                    marker: { color: '#ff7300' },
                    yaxis: 'y2',
                    hovertemplate: '%{y:.2%}',
                  },
                  {
                    x: revenueData.years,
                    y: revenueData.ebitda_margin.map((m) => m / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EBITDA Margin',
                    marker: { color: '#00C49F' },
                    yaxis: 'y2',
                    hovertemplate: '%{y:.2%}',
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Revenue ($)', tickformat: '$,', automargin: true },
                  yaxis2: {
                    title: 'Margins (%)',
                    overlaying: 'y',
                    side: 'right',
                    tickformat: ',.0%',
                  },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '270px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {trafficData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Traffic (LTV vs CAC)
              </Typography>
              <Plot
                data={[
                  {
                    x: trafficData.years,
                    y: trafficData.cac,
                    type: 'bar',
                    name: 'CAC',
                    marker: { color: '#ff7300' },
                  },
                  {
                    x: trafficData.years,
                    y: trafficData.ltv,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'LTV',
                    marker: { color: '#8884d8' },
                  },
                  {
                    x: trafficData.years,
                    y: trafficData.ltv_cac_ratio,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'LTV/CAC Ratio',
                    marker: { color: '#00C49F' },
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Value ($)', tickformat: '$,', automargin: true },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {profitabilityData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Cash Balance Over Time
              </Typography>
              <Plot
                data={[
                  {
                    x: profitabilityData.years,
                    y: profitabilityData.closing_cash_balance,
                    type: 'scatter',
                    mode: 'lines+markers',
                    fill: 'tozeroy',
                    marker: { color: '#82ca9d' },
                    name: 'Closing Cash',
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Cash Balance ($)', tickformat: '$,', automargin: true },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {breakEvenData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Break-Even Analysis
              </Typography>
              <Plot
                data={[
                  {
                    x: breakEvenData.years,
                    y: breakEvenData.break_even_dollars,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Break-Even',
                    marker: { color: '#d62728' },
                  },
                  {
                    x: breakEvenData.years,
                    y: breakEvenData.actual_sales,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Actual Sales',
                    marker: { color: '#2ca02c' },
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Revenue ($)', tickformat: '$,', automargin: true },
                  legend: { orientation: 'h', y: -0.5 },
                }}
               style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {considerationData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Customer Journey Funnel
              </Typography>
              <Plot
                data={[
                  {
                    x: considerationData.years,
                    y: considerationData.weighted_consideration_rate.map((v) => v / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Consideration Rate',
                    marker: { color: '#1f77b4' },
                    yaxis: 'y1',
                    hovertemplate: '%{y:.2%}',
                  },
                  {
                    x: considerationData.years,
                    y: considerationData.consideration_to_conversion,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Conversion',
                    marker: { color: '#2ca02c' },
                    yaxis: 'y2',
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: {
                    title: 'Consideration Rate (%)',
                    tickformat: ',.0%',
                    automargin: true,
                  },
                  yaxis2: {
                    title: 'Conversion Count',
                    overlaying: 'y',
                    side: 'right',
                    automargin: true,
                  },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {marginSafetyData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Margin of Safety
              </Typography>
              <Plot
                data={[
                  {
                    x: marginSafetyData.years,
                    y: marginSafetyData.margin_safety_dollars,
                    type: 'bar',
                    name: 'Margin $',
                    marker: { color: '#8884d8' },
                    yaxis: 'y1',
                  },
                  {
                    x: marginSafetyData.years,
                    y: marginSafetyData.margin_safety_percentage.map((pct) => pct / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Margin %',
                    marker: { color: '#ff7300' },
                    yaxis: 'y2',
                    hovertemplate: '%{y:.2%}',
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Margin ($)', tickformat: '$,', automargin: true },
                  yaxis2: {
                    title: 'Margin (%)',
                    overlaying: 'y',
                    side: 'right',
                    tickformat: ',.0%',
                    automargin: true,
                  },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {cashflowData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Cash Flow Forecast
              </Typography>
              <Plot
                data={[
                  {
                    x: cashflowData.years,
                    y: cashflowData.cash_from_operations,
                    type: 'bar',
                    name: 'Operating CF',
                    marker: { color: '#00C49F' },
                  },
                  {
                    x: cashflowData.years,
                    y: cashflowData.cash_from_investing,
                    type: 'bar',
                    name: 'Investing CF',
                    marker: { color: '#FF8042' },
                  },
                  {
                    x: cashflowData.years,
                    y: cashflowData.net_cash_flow,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Net CF',
                    marker: { color: '#0000FF' },
                  },
                ]}
                layout={{
                  ...baseLayout,
                  barmode: 'stack',
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Cash Flow ($)', tickformat: '$,', automargin: true },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {profitMarginData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Profitability Margin Trends
              </Typography>
              <Plot
                data={[
                  {
                    x: profitMarginData.years,
                    y: profitMarginData.gross_margin.map((m) => m / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Gross Margin',
                    marker: { color: '#00C49F' },
                  },
                  {
                    x: profitMarginData.years,
                    y: profitMarginData.ebitda_margin.map((m) => m / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EBITDA Margin',
                    marker: { color: '#ff7300' },
                  },
                  {
                    x: profitMarginData.years,
                    y: profitMarginData.net_profit_margin.map((m) => m / 100),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Net Profit Margin',
                    marker: { color: '#8884d8' },
                  },
                ]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: 'Year' },
                  yaxis: { title: 'Margin (%)', tickformat: ',.0%', automargin: true },
                  legend: { orientation: 'h', y: -0.5 },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}

          {waterfallData && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                Profit Bridge Analysis
              </Typography>
              <Plot
                data={[{
                  x: waterfallData.categories,
                  y: waterfallData.values,
                  type: 'waterfall',
                  measure: waterfallData.measures,
                  textposition: 'outside',
                  texttemplate: '%{y:$,}',
                  connector: { line: { color: 'rgb(63, 63, 63)' } },
                }]}
                layout={{
                  ...baseLayout,
                  xaxis: { title: '' },
                  yaxis: { title: 'Amount ($)', tickformat: '$,', automargin: true },
                }}
                style={{ width: '100%', height: '250px' }}
                config={plotConfig}
              />
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
    </div>
  );
}
