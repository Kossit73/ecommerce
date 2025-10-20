// src/components/FinancialSchedulesCard.jsx

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Autocomplete,
  TextField,
  Paper,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
} from '@mui/material';
import BASE_URL from '../../config';

// These strings must match exactly the ScheduleType enum values on the backend:
export const SCHEDULE_OPTIONS = [
  'Income Statement',
  'Balance Sheet',
  'Cash Flow Statement',
  'Capital Assets',
  'Valuation',
  'Customer Metrics',
  'Debt Payment Schedule',
];

// For each schedule, list the exact “snake-case” keys that arrive from the backend JSON.
// We will only render these columns (in this order).
export const COLUMNS_BY_SCHEDULE = {
  'Income Statement': [
    'Year',
    'Net_Revenue',
    'Gross_Profit',
    'EBITDA',
    'Net_Income',
    'Total_Orders',
  ],
  'Cash Flow Statement': [
    'Year',
    'Cash_from_Operations',
    'Cash_from_Investing',
    'Cash_from_Financing',
    'Net_Cash_Flow',
  ],
  'Balance Sheet': [
    'Year',
    'Total_Assets',
    'Total_Liabilities',
    'Total_Equity',
    'Balance_Sheet_Check',
  ],
  'Capital Assets': [
    'Year',
    'Total_Opening_Balance',
    'Total_Additions',
    'Total_Depreciation',
    'Total_Closing_Balance',
  ],
  'Valuation': [
    'Year',
    'EBIT',
    'Unlevered_FCF',
    'PV_of_FCF',
    'Total_Enterprise_Value',
  ],
  'Customer Metrics': [
    'Year',
    'CAC',
    'Contribution_Margin_Per_Order',
    'LTV',
    'LTV_CAC_Ratio',
    'Payback_Orders',
  ],
  'Debt Payment Schedule': [
    'Year',
    'Principal',
    'Interest',
    'Payment',
  ],
  // “Debt Payment Schedule” is special, see logic below:

};

// Turn a snake_case key into a human‐readable header ("Net_Revenue" → "Net Revenue", etc.)
export const humanize = (rawKey) =>
  rawKey
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');

export default function FinancialSchedulesCard({
  initialSelectedSchedules = ['Income Statement'],
  title = 'Financial Schedules',
  allowSelection = true,
}) {
  const [selectedSchedules, setSelectedSchedules] = useState(initialSelectedSchedules);
  const [fetchedSchedules, setFetchedSchedules] = useState([]); // will hold payload.schedules from API
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const joinedInitial = initialSelectedSchedules.join('|');

  useEffect(() => {
    setSelectedSchedules(initialSelectedSchedules);
  }, [joinedInitial]);

  useEffect(() => {
    async function fetchSchedules() {
      if (selectedSchedules.length === 0) {
        setFetchedSchedules([]);
        return;
      }
      setLoading(true);
      setErrorMsg('');
      try {
        const params = new URLSearchParams();
        selectedSchedules.forEach((sch) => {
          params.append('schedules', sch);
        });
        const url = `${BASE_URL}/financial_schedules?${params.toString()}`;
        const resp = await fetch(url);
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(`Error ${resp.status}: ${txt}`);
        }
        const payload = await resp.json();
        // payload.schedules is an array of { schedule: string, data: [...] }
        setFetchedSchedules(payload.schedules || []);
      } catch (err) {
        console.error('Error fetching financial schedules:', err);
        setErrorMsg(err.message || 'Unknown error');
        setFetchedSchedules([]);
      } finally {
        setLoading(false);
      }
    }
    fetchSchedules();
  }, [selectedSchedules]);

  return (
    <Card sx={{ mx: 4, my: 3, borderRadius: 0 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {title}
        </Typography>

        {allowSelection && (
          <Box mb={2} width="50%">
            <Autocomplete
              multiple
              options={SCHEDULE_OPTIONS}
              value={selectedSchedules}
              onChange={(_, newValue) => setSelectedSchedules(newValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Select Schedules to View"
                  placeholder="Schedules"
                  size="small"
                />
              )}
            />
          </Box>
        )}

        {loading && <Typography>Loading schedules…</Typography>}
        {errorMsg && <Typography color="error">{errorMsg}</Typography>}

        {!loading &&
          !errorMsg &&
          // For each scheduleBlock returned from the backend, render a paper‐wrapped Table
          fetchedSchedules.map((scheduleBlock, idx) => {
            const { schedule, data } = scheduleBlock;

            // If the backend returned no rows, show “No data available”
            if (!Array.isArray(data) || data.length === 0) {
              return (
                <Box key={idx} mb={4}>
                  <Typography variant="h6">{schedule}</Typography>
                  <Typography color="text.secondary">No data available.</Typography>
                </Box>
              );
            }

            // If this is NOT “Debt Payment Schedule”, we simply render a flat table of data,
            // pulling out only the keys listed in COLUMNS_BY_SCHEDULE[schedule].
            if (schedule !== 'Debt Payment Schedule') {
              // pick out the exact columns
              const keysToShow = COLUMNS_BY_SCHEDULE[schedule] || [];

              return (
                <Box key={idx} mb={6}>
                  <Typography variant="h6" gutterBottom>
                    {schedule}
                  </Typography>
                  <Paper variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          {keysToShow.map((rawKey) => (
                            <TableCell
                              key={rawKey}
                              sx={{
                                fontWeight: 'bold',
                                minWidth: rawKey === 'Year' ? 80 : 120,
                              }}
                            >
                              {humanize(rawKey)}
                            </TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {data.map((rowObj, rowIdx) => (
                          <TableRow key={rowIdx}>
                            {keysToShow.map((rawKey) => {
                              const cellValue = rowObj[rawKey];
                              let display;
                              if (cellValue == null) {
                                display = '–';
                              } else if (rawKey === 'Year') {
                                display = String(cellValue);
                              } else if (typeof cellValue === 'number') {
                                display = cellValue.toFixed(2);
                              } else {
                                display = String(cellValue);
                              }
                              return <TableCell key={rawKey}>{display}</TableCell>;
                            })}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Paper>
                </Box>
              );
            }

            // ───────────────────────────────────────────────────────────────────
            // If we have “Debt Payment Schedule”, the backend’s `data` array looks like:
            //
            //    data = [
            //      { Year: 2015, Schedules: [ { Debt_Name, Amount, Interest_Rate, Duration, Schedule: [ { Year, Principal, Interest, Payment }, … ] }, … ] },
            //      { Year: 2016, Schedules: [ … ] },
            //      { Year: 2017, Schedules: [ … ] },
            //      … 
            //    ]
            //
            // We want to flatten out each inner “Schedule” array so that each table row is:
            //    Year | Principal | Interest | Payment
            //
            // We will loop over data → for each debtYear → for each debtDetail in debtYear.Schedules → for each row in debtDetail.Schedule
            // and render exactly four columns: Year, Principal, Interest, Payment.
            // ───────────────────────────────────────────────────────────────────

            // Build one big list of rows: each row = { Year, Principal, Interest, Payment }
            const flattenedRows = [];
            data.forEach((debtYearObj) => {
              const thisYear = debtYearObj.Year;
              const schedulesArray = debtYearObj.Schedules;
              if (Array.isArray(schedulesArray)) {
                schedulesArray.forEach((debtDetail) => {
                  // debtDetail.Schedule is itself an array of { Year, Principal, Interest, Payment }
                  if (Array.isArray(debtDetail.Schedule)) {
                    debtDetail.Schedule.forEach((schedRow) => {
                      flattenedRows.push({
                        Year: thisYear,
                        Principal: schedRow.Principal,
                        Interest: schedRow.Interest,
                        Payment: schedRow.Payment,
                      });
                    });
                  }
                });
              }
            });

            // Now render a table with exactly those four columns:
            const debtKeys = COLUMNS_BY_SCHEDULE['Debt Payment Schedule'];

            return (
              <Box key={idx} mb={6}>
                <Typography variant="h6" gutterBottom>
                  {schedule}
                </Typography>
                <Paper variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        {debtKeys.map((rawKey) => (
                          <TableCell
                            key={rawKey}
                            sx={{
                              fontWeight: 'bold',
                              minWidth: rawKey === 'Year' ? 80 : 120,
                            }}
                          >
                            {humanize(rawKey)}
                          </TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {flattenedRows.map((rowObj, rowIdx) => (
                        <TableRow key={rowIdx}>
                          {debtKeys.map((rawKey) => {
                            const cellValue = rowObj[rawKey];
                            let display;
                            if (cellValue == null) {
                              display = '–';
                            } else if (rawKey === 'Year') {
                              display = String(cellValue);
                            } else if (typeof cellValue === 'number') {
                              display = cellValue.toFixed(2);
                            } else {
                              display = String(cellValue);
                            }
                            return <TableCell key={rawKey}>{display}</TableCell>;
                          })}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              </Box>
            );
          })}
      </CardContent>
    </Card>
  );
}
