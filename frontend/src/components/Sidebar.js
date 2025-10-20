// src/components/Sidebar.js
import React, { useState } from 'react';
import {
  Drawer,
  IconButton,
  Typography,
  Box,
  Collapse,
  RadioGroup,
  FormControlLabel,
  Radio,
  Input,
  Select,
  MenuItem,
  Slider,
} from '@mui/material';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import SidebarSlider from './SidebarSlider';
import '../css/Sidebar.css';


const Sidebar = ({
  selected,
  setSelected,
  hideAnalysisControls,
  onLoadExisting,
  onStartNew,
  onUploadFile,
  selectedFileName,
  discountRate,
  setDiscountRate,
  wacc,
  setWacc,
  growthRate,
  setGrowthRate,
  taxRate,
  setTaxRate,
  inflationRate,
  setInflationRate,
  laborRateIncrease,
  setLaborRateIncrease,
  scenario,
  setScenario,
  analysisPeriod,
  setAnalysisPeriod,
  startYear,  
  setStartYear,
  forecastYears,
  setForecastYears,
  onDeleteExistingFile
}) => {
  const [collapsed, setCollapsed] = useState(false);
 
  const currentYear = new Date().getFullYear();
  const maxYear = currentYear + 10;

  const toggleSidebar = () => setCollapsed(c => !c);

  const handleRadioChange = async (e) => {
  const sel = e.target.value;

  // ✅ Check directly what value we're switching FROM and TO
  if (selected === 'loadExisting' && sel === 'startNew') {
    const confirmDelete = window.confirm(
      "An existing file is loaded. Do you want to delete it and start fresh?"
    );
    if (!confirmDelete) return;

    if (typeof onDeleteExistingFile === 'function') {
      console.log("Calling onDeleteExistingFile()");
      await onDeleteExistingFile();
    }
  }

  // ✅ Finally update state
  setSelected(sel);

  // ✅ Fire appropriate callback
  if (sel === 'loadExisting') {
    onLoadExisting?.();
  } else if (sel === 'startNew') {
    onStartNew?.(startYear);
  }
};


  const handleFileChange = e => {
    if (!e.target.files.length) return;
    onUploadFile(e.target.files[0]);
  };

  const sliders = [
    { label: 'Discount Rate (%)', value: discountRate, setValue: setDiscountRate, max: 30 },
    { label: 'WACC (%)', value: wacc, setValue: setWacc, max: 20 },
    { label: 'Growth Rate (%)', value: growthRate, setValue: setGrowthRate, max: 10 },
    { label: 'Tax Rate (%)', value: taxRate, setValue: setTaxRate, max: 50 },
    { label: 'Inflation Rate (%)', value: inflationRate, setValue: setInflationRate, max: 10 },
    { label: 'Labor Increase (%)', value: laborRateIncrease, setValue: setLaborRateIncrease, max: 20 },
    {
        label: 'Forecast Years',
        value: forecastYears,
        setValue: (val) => {
          if (val !== forecastYears) {
            setForecastYears(val);
          }
        },
        max: 20,
        min: 1,
        marks: true,
      },
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: collapsed ? 60 : 320,
        '& .MuiDrawer-paper': { width: collapsed ? 60 : 320 },
      }}
    >
      <Box p={2} display="flex" justifyContent="space-between" alignItems="center">
        {!collapsed && <Typography variant="h6">Ecom Dashboard</Typography>}
        <IconButton onClick={toggleSidebar} size="small">
          {collapsed ? (
            <ArrowForwardIosIcon fontSize="small" />
          ) : (
            <ArrowBackIosNewIcon fontSize="small" />
          )}
        </IconButton>
      </Box>

      <Collapse in={!collapsed}>
        <Box p={2}>
          <Typography variant="subtitle2">File Management</Typography>
          <RadioGroup value={selected} onChange={handleRadioChange}>
            <FormControlLabel value="loadExisting" control={<Radio />} label="Load Existing" />
            <FormControlLabel value="startNew" control={<Radio />} label="Start New" />
            {/* <FormControlLabel value="file" control={<Radio />} label="Upload File" /> */}
          </RadioGroup>

          {selected === 'startNew' && (
            <>
            <Box my={1}>
                            <Typography>Starting Year</Typography>
                            <Input
                              type="number"
                              value={startYear}
                              onChange={e => {
                                const y = Number(e.target.value);
                                setStartYear(y);
                                setAnalysisPeriod([y, y + (forecastYears+1)]);
                              }}
                              fullWidth
                            />
                          </Box>
                       <Box my={2}>
                                       <Typography>Forecast Years</Typography>
                                       <Slider
                                         value={forecastYears}
                                         onChange={(_, val) => setForecastYears(parseInt(val))}
                                         valueLabelDisplay="auto"
                                         min={1}
                                         max={20}
                                         step={1}
                                         marks={[{ value: 5 }, { value: 10 }, { value: 15 }, { value: 20 }]}
                                       />
                                     </Box>
                          </>
          )}

          {selected === 'file' && (
            <Box mt={1}>
              <Input type="file" onChange={handleFileChange} fullWidth />
              {selectedFileName && (
                <Typography variant="caption" mt={1} display="block">
                  Loaded: {selectedFileName}
                </Typography>
              )}
            </Box>
          )}

          {!hideAnalysisControls && (
            <>
              <Typography variant="subtitle2" mt={3}>
                Analysis Settings
              </Typography>
              {sliders.map((s, i) => (
                <SidebarSlider key={i} {...s} />
              ))}

              <Typography variant="subtitle2" mt={3}>
                Scenario
              </Typography>
              <Select
                fullWidth
                size="small"
                value={scenario}
                onChange={e => setScenario(e.target.value)}
                sx={{ mb: 2 }}
              >
                <MenuItem value="Base Case">Base Case</MenuItem>
                <MenuItem value="Best Case">Best Case</MenuItem>
                <MenuItem value="Worst Case">Worst Case</MenuItem>
              </Select>

              <Typography variant="subtitle2" mt={3}>
                Analysis Period
              </Typography>
              <Slider
                              value={
                                Array.isArray(analysisPeriod) && analysisPeriod.length === 2
                                  ? analysisPeriod
                                  : [startYear, startYear + (forecastYears+1)]
                              }
                              onChange={(_, newVal) => setAnalysisPeriod(newVal)}
                              valueLabelDisplay="auto"
                              min={startYear}
                              max={startYear + forecastYears+1}
                              step={1}
                              marks={[
                                { value: startYear, label: String(startYear) },
                                { value: startYear + (forecastYears+1), label: String(startYear + (forecastYears+1)) },
                              ]}
                            />
            </>
          )}
        </Box>
      </Collapse>
    </Drawer>
  );
};

export default Sidebar;
