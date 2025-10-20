// src/components/MetricsAndExportCard.jsx

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Paper,
  Typography,
  TextField,
  Button,
  Grid
} from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';
import BASE_URL from '../../config';

export default function MetricsAndExportCard() {
  // ─── State for Operational Metrics ───────────────────────────────────────
  const [opRows, setOpRows] = useState([]);
  const [opLoading, setOpLoading] = useState(true);
  const [opError, setOpError] = useState("");

  // ─── State for Customer Metrics & IRR ────────────────────────────────────
  const [custRows, setCustRows] = useState([]);
  const [irr, setIrr] = useState(null);
  const [custLoading, setCustLoading] = useState(true);
  const [custError, setCustError] = useState("");

  useEffect(() => {
    // Fetch operational metrics
    async function fetchOperationalMetrics() {
      setOpLoading(true);
      setOpError("");
      try {
        const resp = await fetch(`${BASE_URL}/operational_metrics`, {
          credentials: "include",
        });
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(`Operational metrics fetch failed: ${resp.status} ${text}`);
        }
        const data = await resp.json();
        // data.metrics is an array of { metric: string, current: number }
        // We want rows: { id, metric, current }
        const rows = data.metrics.map((m, idx) => {
          // Determine if current should be displayed with a "%" suffix:
          // Any metric containing "Growth", "Margin", or "Return" → percentage
          const isPct = /Growth|Margin|Return/i.test(m.metric);
          const displayValue = isPct
            ? `${m.current.toFixed(1)}%`
            : m.current.toString();
          return {
            id: idx,
            metric: m.metric,
            current: displayValue,
          };
        });
        setOpRows(rows);
      } catch (err) {
        console.error(err);
        setOpError(err.message || "Unknown error");
      } finally {
        setOpLoading(false);
      }
    }

    // Fetch customer metrics
    async function fetchCustomerMetrics() {
      setCustLoading(true);
      setCustError("");
      try {
        const resp = await fetch(`${BASE_URL}/customer_metrics`, {
          credentials: "include",
        });
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(`Customer metrics fetch failed: ${resp.status} ${text}`);
        }
        const data = await resp.json();
        // data.irr is a number (e.g. 50.45)
        // data.metrics is an array of CustomerMetric:
        //   { Year, NPV, CAC, Contribution_Margin_Per_Order, LTV, LTV_CAC_Ratio, Payback_Orders, Burn_Rate }
        setIrr(data.irr);

        // We only have columns for Year, NPV, CAC, Contribution Margin Per Order, and LTV
        // Format each numeric value appropriately.
        const rows = data.metrics.map((m, idx) => ({
          id: idx,
          year: Number(m.Year),
          npv: m.NPV != null ? `$${Number(m.NPV).toLocaleString()}` : "-",
          cac: m.CAC != null ? `$${Number(m.CAC).toFixed(2)}` : "-",
          contribution: m.Contribution_Margin_Per_Order != null
            ? `$${Number(m.Contribution_Margin_Per_Order).toFixed(2)}`
            : "-",
          ltv: m.LTV != null ? `$${Number(m.LTV).toFixed(2)}` : "-",
        }));
        setCustRows(rows);
      } catch (err) {
        console.error(err);
        setCustError(err.message || "Unknown error");
      } finally {
        setCustLoading(false);
      }
    }

    fetchOperationalMetrics();
    fetchCustomerMetrics();
  }, []);

  // ─── Columns definitions ─────────────────────────────────────────────────
  const opColumns = [
    { field: "metric", headerName: "Metric", flex: 1 },
    { field: "current", headerName: "Current", width: 120 },
  ];

  const custColumns = [
    { field: "year", headerName: "Year", width: 100 },
    { field: "npv", headerName: "NPV", flex: 1 },
    { field: "cac", headerName: "CAC", flex: 1 },
    {
      field: "contribution",
      headerName: "Contribution Margin Per Order",
      flex: 1.5,
    },
    { field: "ltv", headerName: "LTV", flex: 1 },
  ];

  return (
    <Card sx={{ mx: 6, my: 4, borderRadius: 0 }}>
      <CardContent>
       
          {/* ─── Operational Metrics ────────────────────────────────────────────── */}
          
            <Typography variant="h5" gutterBottom>
              Operational Metrics
            </Typography>
            {opLoading ? (
  <Typography>Loading operational metrics…</Typography>
) : opError ? (
  <Typography color="error">{opError}</Typography>
) : (
  <Paper variant="outlined" sx={{ width: 450, overflowX: 'auto' }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell sx={{ fontWeight: 'bold' }}>Metric</TableCell>
          <TableCell sx={{ fontWeight: 'bold' }}>Current</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {opRows.map((row) => (
          <TableRow key={row.id}>
            <TableCell>{row.metric}</TableCell>
            <TableCell>{row.current}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </Paper>
)}
      

        <Box my={6} />

       <Grid >
        
          {/* ─── Customer Metrics ──────────────────────────────────────────────── */}
          <Grid item xs={12} md={10}>
            <Typography variant="h5" gutterBottom>
              IRR Metrics
            </Typography>

           {custLoading ? (
  <Typography>Loading customer metrics…</Typography>
) : custError ? (
  <Typography color="error">{custError}</Typography>
) : (
  <>
    <Typography variant="h3" sx={{ mb: 2 }}>
      {`${irr != null ? irr.toFixed(2) : "0.00"}%`}
    </Typography>

    <Paper variant="outlined" sx={{ width: '100%', overflowX: 'auto' }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 'bold' }}>Year</TableCell>
            <TableCell sx={{ fontWeight: 'bold' }}>NPV</TableCell>
            <TableCell sx={{ fontWeight: 'bold' }}>CAC</TableCell>
            <TableCell sx={{ fontWeight: 'bold' }}>Contribution Margin Per Order</TableCell>
            <TableCell sx={{ fontWeight: 'bold' }}>LTV</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {custRows.map((row) => (
            <TableRow key={row.id}>
              <TableCell>{row.year}</TableCell>
              <TableCell>{row.npv}</TableCell>
              <TableCell>{row.cac}</TableCell>
              <TableCell>{row.contribution}</TableCell>
              <TableCell>{row.ltv}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  </>
)}
          </Grid>
       </Grid>
      </CardContent>
    </Card>
  );
}
