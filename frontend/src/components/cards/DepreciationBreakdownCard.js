// src/components/DepreciationBreakdownCard.jsx

import React from "react";
import {
  Box,
  Typography,
  Button,
  TextField,
  Card,
  CardContent,
} from "@mui/material";
import CustomNumericField from "../CustomNumericField";

const DepreciationBreakdownCard = ({ data, onUpdate, onAdd, onRemove }) => {
  // Compute NBV = amount * (1 − rate/100)
  const getNetBookValue = (amount, rate) =>
    (amount * (1 - rate / 100)).toFixed(2);


 // Now sum (amount * rate/100) for each asset:
 const totalDepreciation = data
   .reduce((sum, a) => sum + Number(a.amount || 0) * (Number(a.rate || 0) / 100), 0)
   .toFixed(2);

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Depreciation Breakdown
        </Typography>

        {data.map(({ id, name, amount, rate }) => (
          <Box key={id} display="flex" alignItems="center" gap={1} mb={1}>
            <TextField
              label="Asset Name"
              value={name}
              size="small"
              sx={{ width: "25%" }}
              onChange={(e) => onUpdate(id, "name", e.target.value)}
            />

            <Box width="15%">
              <CustomNumericField
                label="Amount"
                value={amount}
                onChange={(val) => onUpdate(id, "amount", val)}
                step={0.01}
              />
            </Box>

            <Box width="15%">
              <CustomNumericField
                label="Rate (%)"
                value={rate}
                onChange={(val) => onUpdate(id, "rate", val)}
                step={0.1}
              />
            </Box>

            <Box width="20%">
              <CustomNumericField
                label="NBV"
                value={parseFloat(getNetBookValue(amount, rate))}
                disabled
              />
            </Box>

            <Button
              onClick={() => onRemove(id)}
              variant="outlined"
              size="small"
              color="error"
            >
              Remove
            </Button>
          </Box>
        ))}

        <Button onClick={onAdd} variant="contained" size="small" sx={{ mt: 1 }}>
          Add Asset
        </Button>

        {/* Show the new total depreciation: */}
        <Box mt={2} width="40%">
          <CustomNumericField
            label="Total Depreciation"
            value={parseFloat(totalDepreciation)}
            disabled
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default DepreciationBreakdownCard;
