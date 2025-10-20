// src/components/cards/MarketingExpensesCard.jsx

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import CustomNumericField from '../CustomNumericField';
import '../../css/MarketingExpensesCard.css';

const benefitTypes = ['Pension', 'Medical Insurance', 'Child Benefit', 'Car Benefit'];
const executiveRoles = ['CEO', 'COO', 'CFO', 'Director of HR', 'CIO'];
const staffRoles = ['Direct Staff', 'Indirect Staff', 'Part-Time Staff'];

const MarketingExpensesCard = ({ yearData, onUpdateField }) => {
  if (!yearData) return null;

  // 1) Compute staff wages total (hours × number × rate)
  const wagesTotal = staffRoles.reduce((sum, role) => {
    const hours = Number(yearData[`${role} Hours`] || 0);
    const number = Number(yearData[`${role} Number`] || 0);
    const rate = Number(yearData[`${role} Rate`] || 0);
    return sum + hours * number * rate;
  }, 0);

  // 2) Compute executive salaries total
  const execTotal = executiveRoles.reduce((sum, role) => {
    return sum + Number(yearData[`${role} Salary`] || 0);
  }, 0);

  // 3) Compute total staff count
  const totalStaffCount = staffRoles.reduce((sum, role) => {
    return sum + Number(yearData[`${role} Number`] || 0);
  }, 0);

  // 4) Compute each benefit’s total cost and accumulate
  let benefitsTotal = 0;
  const benefitTotalsMap = {};
  benefitTypes.forEach((benefit) => {
    const costPerStaff = Number(yearData[`${benefit} Cost`] || 0);
    const thisTotal = costPerStaff * totalStaffCount;
    benefitTotalsMap[benefit] = thisTotal;
    benefitsTotal += thisTotal;
  });

  // 5) Total compensation (wages + exec + benefits)
  const totalCompensation = wagesTotal + execTotal + benefitsTotal;

  const handleUpdate = (field) => (value) => {
    onUpdateField(field, isNaN(value) ? 0 : value);
  };

  return (
    <Card className="marketing-card">
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Marketing Expenses
        </Typography>
        <Box className="marketing-content">
          {/* LEFT SIDE */}
          <Box className="marketing-left">
            <Typography variant="subtitle1">General Marketing Costs</Typography>
            <CustomNumericField
              label="Email Cost per Click"
              value={yearData['Email CPC'] || 0}
              onChange={handleUpdate('Email CPC')}
              step={0.01}
            />
            <CustomNumericField
              label="Organic Search CPC"
              value={yearData['Organic CPC'] || 0}
              onChange={handleUpdate('Organic CPC')}
              step={0.01}
            />
            <CustomNumericField
              label="Paid Search CPC"
              value={yearData['Paid CPC'] || 0}
              onChange={handleUpdate('Paid CPC')}
              step={0.01}
            />
            <CustomNumericField
              label="Affiliates CPC"
              value={yearData['Affiliates CPC'] || 0}
              onChange={handleUpdate('Affiliates CPC')}
              step={0.01}
            />
            <CustomNumericField
              label="Freight/Shipping per Order"
              value={yearData['Freight per Order'] || 0}
              onChange={handleUpdate('Freight per Order')}
              step={0.01}
            />
            <CustomNumericField
              label="Labor/Handling per Order"
              value={yearData['Labor per Order'] || 0}
              onChange={handleUpdate('Labor per Order')}
              step={0.01}
            />
            <CustomNumericField
              label="General Warehouse Rent"
              value={yearData['Warehouse Rent'] || 0}
              onChange={handleUpdate('Warehouse Rent')}
              step={0.01}
            />

            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle1">Other Expenses</Typography>
            <CustomNumericField
              label="Other ($)"
              value={yearData['Other'] || 0}
              onChange={handleUpdate('Other')}
              step={0.01}
            />
            <CustomNumericField
              label="Interest ($)"
              value={yearData['Interest'] || 0}
              onChange={handleUpdate('Interest')}
              step={0.01}
            />
            <CustomNumericField
              label="Tax Rate (%)"
              value={yearData['Tax Rate'] || 0}
              onChange={handleUpdate('Tax Rate')}
              step={0.01}
            />
          </Box>

          {/* RIGHT SIDE */}
          <Box className="marketing-right">
            {/* Staff Costs */}
            <Typography variant="subtitle1">Salaries, Wages & Benefits Breakdown</Typography>
            <Typography variant="body2" className="section-title">Staff Costs</Typography>
            <Box className="staff-costs-header">
              <span>Category</span>
              <span>Hours/Year</span>
              <span>Number</span>
              <span>Hourly Rate ($)</span>
            </Box>
            {staffRoles.map((role) => (
              <Box className="staff-costs-row" key={role}>
                <span>{role}</span>
                <CustomNumericField
                  label=""
                  value={yearData[`${role} Hours`] || 0}
                  onChange={handleUpdate(`${role} Hours`)}
                  step={100}
                />
                <CustomNumericField
                  label=""
                  value={yearData[`${role} Number`] || 0}
                  onChange={handleUpdate(`${role} Number`)}
                  step={1}
                />
                <CustomNumericField
                  label=""
                  value={yearData[`${role} Rate`] || 0}
                  onChange={handleUpdate(`${role} Rate`)}
                  step={1}
                />
              </Box>
            ))}

            {/* Executive Salaries */}
            <Typography variant="body2" className="section-title" sx={{ mt: 2 }}>
              Executive Salaries
            </Typography>
            <Box className="exec-salary-header">
              <span>Position</span>
              <span>Annual Salary ($)</span>
            </Box>
            {executiveRoles.map((role) => (
              <Box className="exec-salary-row" key={role}>
                <span>{role}</span>
                <CustomNumericField
                  label=""
                  value={yearData[`${role} Salary`] || 0}
                  onChange={handleUpdate(`${role} Salary`)}
                  step={1000}
                />
              </Box>
            ))}

            {/* Benefits */}
            <Typography variant="body2" className="section-title" sx={{ mt: 2 }}>
              Benefits
            </Typography>
            <Box className="benefits-header">
              <span>Type</span>
              <span>Cost/Staff ($)</span>
              <span>Total Cost ($)</span>
            </Box>
            {benefitTypes.map((benefit) => {
              const costPerStaff = Number(yearData[`${benefit} Cost`] || 0);
              const totalCost = benefitTotalsMap[benefit] || 0;
              return (
                <Box className="benefits-row" key={benefit}>
                  <span>{benefit}</span>
                  <CustomNumericField
                    label=""
                    value={costPerStaff}
                    onChange={handleUpdate(`${benefit} Cost`)}
                    step={10}
                  />
                  <CustomNumericField
                    label=""
                    value={parseFloat(totalCost.toFixed(2))}
                    disabled
                  />
                </Box>
              );
            })}

            {/* Salaries, Wages & Benefits Total */}
            <Box display="flex" mt={2} alignItems="center">
              <Typography variant="subtitle2" sx={{ flexGrow: 1 }}>
                Salaries, Wages & Benefits
              </Typography>
              <CustomNumericField
                label=""
                value={parseFloat(totalCompensation.toFixed(2))}
                disabled
              />
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MarketingExpensesCard;
