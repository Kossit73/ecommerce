import React from 'react';
import { Box, Typography, Button, TextField ,Card,CardContent} from '@mui/material';
import CustomNumericField from '../CustomNumericField';

const DebtIssuedCard = ({ data, onUpdate, onAdd, onRemove }) => {
  const totalDebt = data.reduce((sum, d) => sum + Number(d.amount), 0).toFixed(2);

  const handleChange = (id, field, value) => {
    onUpdate(id, field, value);
  };

  return (
    <Card className="marketing-card">
          <CardContent>
    <Box mt={4}>
      <Typography variant="h6" gutterBottom>Debt Issued Breakdown</Typography>

      <Box mb={2} width="50%">
        <CustomNumericField
          label="Total Debt Issued"
          value={parseFloat(totalDebt)}
          onChange={() => {}}
          disabled
        />
      </Box>

      {/* Table Header */}
      <Box display="flex" fontWeight="bold" mb={1}>
        <Box width="25%">Debt Name</Box>
        <Box width="20%">Amount ($)</Box>
        <Box width="20%">Interest Rate (%)</Box>
        <Box width="20%">Duration (Years)</Box>
        <Box width="15%">Action</Box>
      </Box>

      {/* Debt Rows */}
      {data.map(({ id, name, amount, interestRate, duration }) => (
        <Box key={id} display="flex" alignItems="center" gap={1} mb={1}>
          <Box width="25%">
            <TextField
              value={name}
              onChange={(e) => handleChange(id, 'name', e.target.value)}
              size="small"
              fullWidth
              variant="outlined"
            />
          </Box>
          <Box width="20%">
            <CustomNumericField
              value={amount}
              onChange={(val) => handleChange(id, 'amount', val)}
              step={1000}
            />
          </Box>
          <Box width="20%">
            <CustomNumericField
              value={interestRate}
              onChange={(val) => handleChange(id, 'interestRate', val)}
              step={0.01}
            />
          </Box>
          <Box width="20%">
            <CustomNumericField
              value={duration}
              onChange={(val) => handleChange(id, 'duration', val)}
              step={1}
            />
          </Box>
          <Box width="15%">
            <Button onClick={() => onRemove(id)} variant="outlined" size="small" color="error">
              Remove
            </Button>
          </Box>
        </Box>
      ))}

      <Box mt={2}>
        <Button variant="contained" onClick={onAdd}>
          Add Debt
        </Button>
      </Box>
    </Box>
    </CardContent>
        </Card>
  );
};

export default DebtIssuedCard;
