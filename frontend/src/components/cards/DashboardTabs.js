import React, { useState } from 'react';
import {
  Box,
  Tabs,
  Tab,
  Typography,
  Divider,
  Button,
} from '@mui/material';
import MetricsAndExportCard from './MetricsAndExportCard';
import FinancialSchedulesCard from './FinancialSchedulesCard';
import DCFValuationCard from './DCFValuationCard';
import DetailedAnalysisCard from './DetailedAnalysisCard';
import AdvancedDecisionToolsCard from './AdvancedDecisionToolsCard';
import DownloadIcon from '@mui/icons-material/Download';
import ExcelJS from 'exceljs';
import { saveAs } from 'file-saver';
import Plotly from 'plotly.js-dist-min'
import BASE_URL from '../../config';
function TabPanel({ children, value, index, ...props }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`dashboard-tabpanel-${index}`}
      aria-labelledby={`dashboard-tab-${index}`}
      {...props}
    >
      {value === index && (
        <Box sx={{ py: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function DashboardTabs({
  discountRate,
  wacc,
  growthRate,
  scenario,
  taxRate,
  selectedSchedules,
  setSelectedSchedules,
  scheduleData,
  scheduleLoading,
  scheduleError,
}) {
  const [tab, setTab] = useState(0);
  const handleChange = (_, newValue) => {
    setTab(newValue);
  };
  const chartNode = document.getElementById('combined-performance-chart');
  console.log("chart:",chartNode )
  function downloadScenarioWorkbook() {
  fetch(`${BASE_URL}/api/ecommerce/export_excel`, {
                 // if you’re using session auth
    headers: { 'Content-Type': 'application/octet-stream' }
  })
    .then(res => {
      if (!res.ok) throw new Error("Failed to export");
      return res.blob();
    })
    .then(blob => saveAs(blob, "All_Scenarios_Report.xlsx"))
    .catch(err => console.error("Export failed:", err));
}
  const handleDownloadReport = async () => {
    const wb = new ExcelJS.Workbook();

    const scheduleMap = {
      'Income Statement': ['Net_Revenue', 'Gross_Profit', 'EBITDA', 'Net_Income', 'Total_Orders'],
      'Balance Sheet': ['Total_Assets', 'Total_Liabilities', 'Total_Equity', 'Balance_Sheet_Check'],
      'Cash Flow Statement': ['Cash_from_Operations', 'Cash_from_Investing', 'Cash_from_Financing', 'Net_Cash_Flow'],
      'Valuation': ['EBIT', 'Unlevered_FCF', 'PV_of_FCF', 'Total_Enterprise_Value'],
      'Customer Metrics': ['CAC', 'Contribution_Margin_Per_Order', 'LTV', 'LTV_CAC_Ratio', 'Payback_Orders']
    };

    scheduleData.forEach(({ schedule, data }) => {
      const keys = scheduleMap[schedule];
      if (!keys || !data.length) return;

      const ws = wb.addWorksheet(schedule);
      const years = data.map(row => row.Year);
      const headers = ['Metric', ...years];

      keys.forEach(metric => {
        const row = [metric, ...data.map(row => row[metric] ?? '')];
        ws.addRow(row);
      });

      ws.insertRow(1, headers);
    });
     try {
      
      // ask Plotly to snapshot that container
      const pngDataUrl = await Plotly.toImage(chartNode, {
        format:  'png',
        width:   800,    // tweak to your desired image size
        height:  400,
      });

      // strip off the prefix (“data:image/png;base64,”)
      const base64Image = pngDataUrl.split(',')[1];

      // register it with ExcelJS
      const imageId = wb.addImage({
        base64:   base64Image,
        extension:'png',
      });

      // make a new “Charts” sheet and put the picture at A1
      const wsCharts = wb.addWorksheet('Charts');
      wsCharts.addImage(imageId, {
        tl: { col: 0, row: 0 },
        ext:{ width: 800*0.75, height: 400*0.75 }  // pixels → points
      });
    } catch (err) {
      console.warn("Could not snapshot CombinedPerformanceCard:", err);
    }
    const buf = await wb.xlsx.writeBuffer();
    saveAs(new Blob([buf], { type: 'application/octet-stream' }), `Ecommerce_${scenario.replace(/\s+/g, '_')}_Financial_Report.xlsx`);
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs
        value={tab}
        onChange={handleChange}
        indicatorColor="secondary"
        textColor="inherit"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
      >
        <Tab label="Key Metrics" />
        <Tab label="Financial Schedules" />
        <Tab label="DCF Analysis" />
        <Tab label="Detailed Analysis" />
        <Tab label="Advanced Tools" />
      </Tabs>

      <Divider />

      <TabPanel value={tab} index={0}>
        <Typography variant="h4" gutterBottom>
          Operational Metrics
        </Typography>
        <MetricsAndExportCard />
        <Box height={32} />
      </TabPanel>

      <TabPanel value={tab} index={1}>
        <Typography variant="h4" gutterBottom>
          Financial Schedules
        </Typography>
        <FinancialSchedulesCard
          selectedSchedules={selectedSchedules}
          setSelectedSchedules={setSelectedSchedules}
          scheduleData={scheduleData}
          loading={scheduleLoading}
          error={scheduleError}
        />
      </TabPanel>

      <TabPanel value={tab} index={2}>
        <Typography variant="h4" gutterBottom>
          DCF Analysis
        </Typography>
        <DCFValuationCard />
      </TabPanel>

      <TabPanel value={tab} index={3}>
        <Typography variant="h4" gutterBottom>
          Detailed Analysis
        </Typography>
        <DetailedAnalysisCard
          discountRate={discountRate}
          wacc={wacc}
          growthRate={growthRate}
          scenario={scenario}
          taxRate={taxRate}
        />
      </TabPanel>

      <TabPanel value={tab} index={4}>
        <Typography variant="h4" gutterBottom>
          Advanced Tools
        </Typography>
        <AdvancedDecisionToolsCard
          discountRate={discountRate}
          wacc={wacc}
          growthRate={growthRate}
          scenario={scenario}
          taxRate={taxRate}
        />
      </TabPanel>

      <Box mt={4} display="flex" justifyContent="flex-end">
        <Button
          variant="contained"
          startIcon={<DownloadIcon />}
          onClick={downloadScenarioWorkbook}
        >
          Download Full Report
        </Button>
      </Box>
    </Box>
  );
}
