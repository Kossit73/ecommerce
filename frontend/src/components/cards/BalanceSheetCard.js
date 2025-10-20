// src/components/cards/BalanceSheetCard.js

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import CustomNumericField from '../CustomNumericField';

/**
 * Props:
 *  - data: {
 *      "Accounts Receivable Days": number,
 *      "Inventory Days": number,
 *      "Accounts Payable Days": number,
 *      "Technology Development": number,
 *      "Office Equipment": number,
 *      "Technology Depreciation Years": number,
 *      "Office Equipment Depreciation Years": number,
 *      "Interest Rate (Default)": number
 *    }
 *  - onUpdateField: (fieldName: string, newValue: number) => void
 */
const BalanceSheetCard = ({ data, onUpdateField }) => {
  // Turn the `data` object into [fieldName, value] pairs, so we can map over them:
  const entries = Object.entries(data);

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box mt={2}>
          <Typography variant="h6" gutterBottom>
            Balance Sheet Inputs
          </Typography>

          {entries.map(([fieldName, fieldValue]) => (
            <CustomNumericField
              key={fieldName}
              label={fieldName}
              value={fieldValue}
              onChange={(newVal) => {
                // Pass the numeric value back up to the parent
                onUpdateField(fieldName, newVal);
              }}
              step={1}
              disabled={false}
              sx={{ mb: 1 }}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default BalanceSheetCard;
