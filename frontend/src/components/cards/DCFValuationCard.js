// src/components/DCFValuationCard.jsx
import React, { useEffect, useState } from 'react';
import { Card, CardContent, Grid, Typography, Box } from '@mui/material';
import Plot from 'react-plotly.js';
import BASE_URL from '../../config';

export default function DCFValuationCard() {
  // ─── State for enterprise/equity values ─────────────────────────────────
  const [enterpriseValueM, setEnterpriseValueM] = useState(0.0);
  const [equityValueM, setEquityValueM] = useState(0.0);

  // ─── State for waterfall chart data ────────────────────────────────────
  const [bridgeCategories, setBridgeCategories] = useState([]); // e.g. ["Present Value of FCF", …]
  const [bridgeValues, setBridgeValues] = useState([]);         // e.g. [18000000, 35000000, -10000000, 43000000]
  const [bridgeMeasures, setBridgeMeasures] = useState([]);     // e.g. ["relative","relative","relative","total"]

  // ─── Loading / error flags ──────────────────────────────────────────────
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");

  // Utility to format a raw number in dollars‐millions with one decimal
  const fmtMillions = (num) => `$${(num / 1e6).toFixed(1)}M`;

  // ─── On mount, fetch both endpoints ─────────────────────────────────────
  useEffect(() => {
    async function fetchDcfData() {
      setLoading(true);
      setErrorMsg("");

      try {
        // 1) Fetch enterprise/equity
        const respVal = await fetch(`${BASE_URL}/dcf_valuation`, {
          credentials: "include",
        });
        if (!respVal.ok) {
          const txt = await respVal.text();
          throw new Error(`dcf_valuation error ${respVal.status}: ${txt}`);
        }
        const { enterprise_value_m, equity_value_m } = await respVal.json();
        setEnterpriseValueM(enterprise_value_m);
        setEquityValueM(equity_value_m);

        // 2) Fetch bridge chart data
        const respBridge = await fetch(
          `${BASE_URL}/dcf_summary_chart_data`,
          {
            credentials: "include",
          }
        );
        if (!respBridge.ok) {
          const txt = await respBridge.text();
          throw new Error(`dcf_summary_chart_data error ${respBridge.status}: ${txt}`);
        }
        const payload = await respBridge.json();
        // payload should be: { status: "success", data: { categories: [...], values: [...], measures: [...] } }
        if (payload.status !== "success" || !payload.data) {
          throw new Error("Unexpected DCF summary chart response");
        }
        const { categories, values, measures } = payload.data;
        setBridgeCategories(categories);
        setBridgeValues(values);
        setBridgeMeasures(measures);
      } catch (err) {
        console.error("Error loading DCF data:", err);
        setErrorMsg(err.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    }

    fetchDcfData();
  }, []);

  // If still loading, show a simple message
  if (loading) {
    return (
      <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            DCF Valuation Summary
          </Typography>
          <Typography>Loading...</Typography>
        </CardContent>
      </Card>
    );
  }

  // If error occurred, render it
  if (errorMsg) {
    return (
      <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            DCF Valuation Summary
          </Typography>
          <Typography color="error">{errorMsg}</Typography>
        </CardContent>
      </Card>
    );
  }

  // ─── Construct the Plotly waterfall trace ────────────────────────────────
  const waterfallTrace = {
    type: "waterfall",
    x: bridgeCategories,
    y: bridgeValues,
    measure: bridgeMeasures,
    text: bridgeValues.map((v) => fmtMillions(v)), // show labels on bars
    textposition: "outside",
    connector: { line: { color: "#333" } },
    increasing: { marker: { color: "#00C49F" } }, // green for positive
    decreasing: { marker: { color: "#D62728" } }, // red for negative
    total: { marker: { color: "#0000FF" } },     // blue for final “total” bar
  };

  const layout = {
    margin: { t: 40, r: 10, l: 120, b: 80 },
    yaxis: {
      title: { text: "Value (Millions)" },
      tickformat: "$,.0f",
    },
    xaxis: {
      tickangle: -30,
      automargin: true,
    },
    title: {
      text: "", // Already have a Typo above, so leave blank here
      font: { size: 14 },
    },
    showlegend: false,
    width: 600,
    height: 450,
  };

  return (
    <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          DCF Valuation Summary
        </Typography>

        <Grid container spacing={4} alignItems="flex-start">
          {/* ─── Left: Enterprise / Equity values ────────────────────────────── */}
          <Grid item xs={12} md={4}>
            <Box mb={3}>
              <Typography variant="subtitle2">Enterprise Value</Typography>
              <Typography variant="h4">{fmtMillions(enterpriseValueM * 1e6)}</Typography>
            </Box>
            <Box>
              <Typography variant="subtitle2">Equity Value</Typography>
              <Typography variant="h4">{fmtMillions(equityValueM * 1e6)}</Typography>
            </Box>
          </Grid>

          {/* ─── Right: Waterfall chart ───────────────────────────────────────── */}
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom>
              DCF Valuation Bridge
            </Typography>
            <Plot
              data={[waterfallTrace]}
              layout={layout}
              config={{ displayModeBar: false }}
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}
