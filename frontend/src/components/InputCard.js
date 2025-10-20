import React from 'react';
import { TextField, Box, Typography } from '@mui/material';

const InputCard = ({ label, value, setValue, type = 'number' }) => {
  return (
    <Box mb={2}>
      <Typography gutterBottom>{label}</Typography>
      <TextField
        fullWidth
        value={value}
        onChange={(e) => setValue(e.target.value)}
        type={type}
        size="small"
      />
    </Box>
  );
};

export default InputCard;

