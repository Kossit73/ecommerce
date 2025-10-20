// src/components/cards/OfficeRentBreakdownCard.js

import React from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Card,
  CardContent,
} from '@mui/material';

const OfficeRentBreakdownCard = ({ data, onUpdate, onAdd, onRemove }) => {
  // data is an array of { id, category, squareMeters, costPerSQM }

  // Compute total = sum(squareMeters * costPerSQM)
  const total = data
    .reduce((sum, cat) => sum + cat.squareMeters * cat.costPerSQM, 0)
    .toFixed(2);

  const [newCategory, setNewCategory] = React.useState('');

  const handleAddCategory = () => {
    const name = newCategory.trim();
    if (name) {
      onAdd(name);
      setNewCategory('');
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box mt={2}>
          <Typography variant="h6" gutterBottom>
            Office Rent Breakdown
          </Typography>

          {/* Table Headers */}
          <Box display="flex" fontWeight="bold" mb={1}>
            <Box width="30%">Category</Box>
            <Box width="30%">Square Meters</Box>
            <Box width="30%">Cost per SQM ($)</Box>
            <Box width="10%" /> {/* for Remove button */}
          </Box>

          {/* Rent Rows */}
          {data.map(({ id, category, squareMeters, costPerSQM }, idx) => {
            // As a fallback, also include `idx` in the key to guarantee uniqueness
            const keyString = id != null ? `${id}-${idx}` : `no-id-${idx}`;

            return (
              <Box
                key={keyString}
                display="flex"
                alignItems="center"
                gap={1}
                mb={1}
              >
                <Box width="30%">
                  <Typography variant="body2" sx={{ pt: 1 }}>
                    {category}
                  </Typography>
                </Box>
                <Box width="30%">
                  <TextField
                    type="number"
                    size="small"
                    value={squareMeters}
                    onChange={(e) =>
                      onUpdate(id, 'squareMeters', parseFloat(e.target.value) || 0)
                    }
                    fullWidth
                  />
                </Box>
                <Box width="30%">
                  <TextField
                    type="number"
                    size="small"
                    value={costPerSQM}
                    onChange={(e) =>
                      onUpdate(id, 'costPerSQM', parseFloat(e.target.value) || 0)
                    }
                    fullWidth
                  />
                </Box>
                <Box width="10%">
                  <Button
                    onClick={() => onRemove(id)}
                    variant="outlined"
                    size="small"
                    color="error"
                  >
                    Remove
                  </Button>
                </Box>
              </Box>
            );
          })}

          {/* Add New Rent Category */}
          <Box display="flex" alignItems="center" gap={2} mt={2}>
            <TextField
              variant="outlined"
              size="small"
              placeholder="Add New Rent Category"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              sx={{
                width: '250px',
                background: '#ffffff',
                input: { color: '#000000' },
              }}
            />
            <Button
              variant="contained"
              size="small"
              onClick={handleAddCategory}
            >
              Add Rent Category
            </Button>
          </Box>

          {/* Total Office Rent (read-only) */}
          <Box mt={3} width="50%">
            <TextField
              label="Office Rent (Total)"
              type="number"
              value={parseFloat(total)}
              size="small"
              InputProps={{ readOnly: true }}
              fullWidth
            />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default OfficeRentBreakdownCard;
