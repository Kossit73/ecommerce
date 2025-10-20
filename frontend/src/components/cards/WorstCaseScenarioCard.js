import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  IconButton,
} from '@mui/material';
import SidebarSlider from '../SidebarSlider';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowUpIcon from '@mui/icons-material/KeyboardArrowUp';

const WorstCaseScenarioCard = ({ data, onChange }) => {
  const [expanded, setExpanded] = useState(false);

  const toggleExpand = () => {
    setExpanded((prev) => !prev);
  };

  return (
    <Card className="marketing-card" sx={{ mb: 3 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Worst Case Scenario Parameters</Typography>
          <IconButton onClick={toggleExpand}>
            {expanded ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
          </IconButton>
        </Box>

        {expanded && (
          <Box display="flex" flexWrap="wrap" gap={3}>
            {/* Left Column */}
            <Box width="48%">
              <SidebarSlider
                label="Conversion Rate Multiplier"
                value={data.conversion_rate_mult}
                setValue={(val) => onChange('conversion_rate_mult', val)}
                min={0.5}
                max={1}
                step={0.01}
              />
              <SidebarSlider
                label="Average Order Value Multiplier"
                value={data.aov_mult}
                setValue={(val) => onChange('aov_mult', val)}
                min={0.5}
                max={1}
                step={0.01}
              />
              <SidebarSlider
                label="Political Risk"
                value={data.political_risk}
                setValue={(val) => onChange('political_risk', val)}
                min={1}
                max={5}
                step={1}
              />
              <SidebarSlider
                label="Environmental Impact"
                value={data.env_impact}
                setValue={(val) => onChange('env_impact', val)}
                min={1}
                max={5}
                step={1}
              />
            </Box>

            {/* Right Column */}
            <Box width="48%">
              <SidebarSlider
                label="COGS Multiplier (%)"
                value={data.cogs_mult}
                setValue={(val) => onChange('cogs_mult', val)}
                min={1}
                max={1.2}
                step={0.01}
              />
              <SidebarSlider
                label="Interest Cost Multiplier"
                value={data.interest_mult}
                setValue={(val) => onChange('interest_mult', val)}
                min={1}
                max={1.5}
                step={0.01}
              />
              <SidebarSlider
                label="Labor Cost Multiplier"
                value={data.labor_mult}
                setValue={(val) => onChange('labor_mult', val)}
                min={1}
                max={1.5}
                step={0.01}
              />
              <SidebarSlider
                label="Material Cost Multiplier"
                value={data.materialCostIncrease}
                setValue={(val) => onChange('materialCostIncrease', val)}
                min={1}
                max={1.5}
                step={0.01}
              />
              <SidebarSlider
                label="Markdown Adjustment"
                value={data.material_mult}
                setValue={(val) => onChange('material_mult', val)}
                min={1}
                max={1.2}
                step={0.01}
              />
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default WorstCaseScenarioCard;
