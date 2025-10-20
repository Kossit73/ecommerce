
import React from 'react';
import { Card, CardContent, Typography, Grid, TextField } from '@mui/material';

const SalariesBenefitsCard = ({ year, yearData, onUpdateField }) => {
  const handleInput = (field) => (e) => {
    onUpdateField(year, field, e.target.value);
  };

  const staffTypes = [
    { label: 'Direct Staff', defaultHours: 2080 },
    { label: 'Indirect Staff', defaultHours: 2080 },
    { label: 'Part-Time Staff', defaultHours: 1040 },
  ];

  const execRoles = ['CEO', 'COO', 'CFO', 'Director of HR', 'CIO'];
  const benefits = ['Pension', 'Medical Insurance', 'Child Benefit', 'Car Benefit'];

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Salaries, Wages & Benefits Breakdown
        </Typography>

        <Typography variant="subtitle1">Staff Costs</Typography>
        {staffTypes.map(({ label }) => (
          <Grid container spacing={2} key={label} alignItems="center">
            <Grid item xs={4}><TextField label={`${label} Hours/Year`} fullWidth type="number" value={yearData[`${label} Hours per Year`] || ''} onChange={handleInput(`${label} Hours per Year`)} /></Grid>
            <Grid item xs={4}><TextField label={`${label} Number`} fullWidth type="number" value={yearData[`${label} Number`] || ''} onChange={handleInput(`${label} Number`)} /></Grid>
            <Grid item xs={4}><TextField label={`${label} Hourly Rate ($)`} fullWidth type="number" value={yearData[`${label} Hourly Rate`] || ''} onChange={handleInput(`${label} Hourly Rate`)} /></Grid>
          </Grid>
        ))}

        <Typography variant="subtitle1" sx={{ mt: 3 }}>Executive Salaries</Typography>
        {execRoles.map((role) => (
          <Grid container spacing={2} key={role}>
            <Grid item xs={12}><TextField label={`${role} Salary`} fullWidth type="number" value={yearData[`${role} Salary`] || ''} onChange={handleInput(`${role} Salary`)} /></Grid>
          </Grid>
        ))}

        <Typography variant="subtitle1" sx={{ mt: 3 }}>Benefits</Typography>
        {benefits.map((benefit) => (
          <Grid container spacing={2} key={benefit}>
            <Grid item xs={6}><TextField label={`${benefit} Cost/Staff`} fullWidth type="number" value={yearData[`${benefit} Cost per Staff`] || ''} onChange={handleInput(`${benefit} Cost per Staff`)} /></Grid>
            <Grid item xs={6}><TextField label={`${benefit} Total Cost`} fullWidth type="number" value={yearData[`${benefit} Total Cost`] || ''} onChange={handleInput(`${benefit} Total Cost`)} /></Grid>
          </Grid>
        ))}

        <Grid container spacing={2} sx={{ mt: 2 }}>
          <Grid item xs={6}><TextField label="Total Benefits" fullWidth type="number" value={yearData['Total Benefits'] || ''} onChange={handleInput('Total Benefits')} /></Grid>
          <Grid item xs={6}><TextField label="Salaries, Wages & Benefits (Total)" fullWidth type="number" value={yearData['Salaries, Wages & Benefits'] || ''} onChange={handleInput('Salaries, Wages & Benefits')} /></Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default SalariesBenefitsCard;
