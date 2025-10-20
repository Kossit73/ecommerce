// src/components/cards/ProfessionalFeesCard.jsx

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  Card,
  CardContent,
} from '@mui/material';
import CustomNumericField from '../CustomNumericField';

const ProfessionalFeesCard = () => {
  // 1) Add a separate state for "Legal Cost"
  const [legalCost, setLegalCost] = useState(0);

  // 2) Keep the existing dynamic categories state
  const [categories, setCategories] = useState([
    { id: 1, name: 'Accounting', Cost: 0 },
    // you can seed other default fee types here if needed
  ]);
  const [newCategory, setNewCategory] = useState('');

  // 3) Handlers for dynamic categories
  const handleChange = (id, field, value) => {
    setCategories(prev =>
      prev.map(cat => (cat.id === id ? { ...cat, [field]: value } : cat))
    );
  };

  const handleRemove = (id) => {
    setCategories(prev => prev.filter(cat => cat.id !== id));
  };

  const handleAddCategory = () => {
    const name = newCategory.trim();
    if (name) {
      setCategories(prev => [
        ...prev,
        { id: Date.now(), name, Cost: 0 },
      ]);
      setNewCategory('');
    }
  };

  // 4) Compute totals: include legalCost + sum of all other categories
  const categoriesTotal = categories.reduce((sum, cat) => sum + Number(cat.Cost || 0), 0);
  const total = (Number(legalCost || 0) + categoriesTotal).toFixed(2);

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Professional Fees Breakdown
        </Typography>

        {/* 5) Dedicated "Legal Cost" field */}
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Typography sx={{ width: '40%' }}>Legal Cost ($)</Typography>
          <Box width="30%">
            <CustomNumericField
              value={legalCost}
              onChange={(val) => setLegalCost(val)}
              step={1}
            />
          </Box>
        </Box>

        {/* Table Headers for dynamic categories */}
        <Box display="flex" fontWeight="bold" mb={1}>
          <Box width="40%">Fee Type</Box>
          <Box width="30%">Cost ($)</Box>
          <Box width="20%" /> {/* empty space for Remove button */}
        </Box>

        {/* Dynamic Fee Rows */}
        {categories.map(({ id, name, Cost }) => (
          <Box key={id} display="flex" alignItems="center" gap={1} mb={1}>
            <Box width="40%">
              <Typography variant="body2" sx={{ pt: 1 }}>
                {name}
              </Typography>
            </Box>
            <Box width="30%">
              <CustomNumericField
                value={Cost}
                onChange={(val) => handleChange(id, 'Cost', val)}
                step={1}
              />
            </Box>
            <Box width="20%">
              <Button
                onClick={() => handleRemove(id)}
                variant="outlined"
                size="small"
                color="error"
              >
                Remove
              </Button>
            </Box>
          </Box>
        ))}

        {/* Add New Fee Type */}
        <Box display="flex" alignItems="center" gap={2} mt={2}>
          <TextField
            variant="outlined"
            size="small"
            placeholder="Add New Fee Type"
            value={newCategory}
            onChange={(e) => setNewCategory(e.target.value)}
            sx={{ width: '250px' }}
          />
          <Button variant="contained" size="small" onClick={handleAddCategory}>
            Add New Fee Type
          </Button>
        </Box>

        {/* Total Professional Fees (includes Legal Cost) */}
        <Box mt={3} width="50%">
          <CustomNumericField
            label="Professional Fees (Total) ($)"
            value={parseFloat(total)}
            disabled
            onChange={() => {}}
          />
        </Box>
      </CardContent>
    </Card>
  );
};

export default ProfessionalFeesCard;
