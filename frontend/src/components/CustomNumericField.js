
import React from 'react';
import {
  TextField,
  InputAdornment,
  IconButton,
  Typography,
  Box,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';

const CustomNumericField = ({
  label,
  value,
  onChange,
  step = 1,
  disabled = false,
  width = '100%', 
  height = '32px',      
  sx,                   
}) => {
  const handleIncrease = () => {
    if (!disabled) onChange(typeof value === 'number' ? value + step : step);
  };
  const handleDecrease = () => {
    if (!disabled) onChange(typeof value === 'number' ? value - step : 0);
  };

  return (
    <Box mb={1} sx={{ width, ...sx }}>
      {label && (
        <Typography
          variant="body2"
          fontWeight={500}
          gutterBottom
          sx={{ lineHeight: 1.2 }}
        >
          {label}
        </Typography>
      )}
      <TextField
        type="number"
        value={value ?? ''}
        onChange={(e) => {
          const val = parseFloat(e.target.value);
          onChange(isNaN(val) ? '' : val);
        }}
        variant="outlined"
        size="small"
        disabled={disabled}
        fullWidth            // will fill the Box’s width
        InputProps={{
          inputProps: {
            step,
            style: {
              MozAppearance: 'textfield',
              padding: '6px 8px',
              fontSize: '0.75rem',
              height,
            },
          },
          endAdornment: !disabled && (
            <InputAdornment position="end">
              <IconButton onClick={handleDecrease} size="small">
                <RemoveIcon fontSize="small" />
              </IconButton>
              <IconButton onClick={handleIncrease} size="small">
                <AddIcon fontSize="small" />
              </IconButton>
            </InputAdornment>
          ),
        }}
        sx={{
          '& input[type=number]::-webkit-outer-spin-button, & input[type=number]::-webkit-inner-spin-button': {
            WebkitAppearance: 'none',
            margin: 0,
          },
          '& input[type=number]': {
            MozAppearance: 'textfield',
          },
          '& .MuiInputBase-root': {
            
            fontSize: '0.75rem',
          },
        }}
      />
    </Box>
  );
};

export default CustomNumericField;
