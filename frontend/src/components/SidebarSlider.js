
import React from 'react';
import { Typography, Slider, Box } from '@mui/material';

const SidebarSlider = ({ label, value, setValue, min = 0, max = 100, step = 0.5 }) => {
  return (
    <Box mb={2}>
      <Typography variant="body2" gutterBottom>{label}</Typography>
      <Slider
        size="small"
        value={value}
        onChange={(e, newVal) => setValue(newVal)}
        min={min}
        max={max}
        step={step}
        valueLabelDisplay="auto"
        sx={{ mt: 1 }}
      />
      <Box display="flex" justifyContent="space-between" mt={-1} px={0.5}>
        <Typography variant="caption">{min}</Typography>
        <Typography variant="caption">{max}</Typography>
      </Box>
    </Box>
  );
};

export default SidebarSlider;
