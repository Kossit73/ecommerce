// src/default.js

export const defaults = {
  // ─── Traffic & Conversion Defaults ─────────────────────────────────────────
  traffic :{"Email Traffic": 0,
  "Organic Search Traffic": 10000,
  "Paid Search Traffic": 30000,
  "Affiliates Traffic": 20000,
  "Email Conversion Rate": 0.06,
  "Organic Search Conversion Rate": 0.02,
  "Paid Search Conversion Rate": 0.022,
  "Affiliates Conversion Rate": 0.04,
  "Average Item Value": 250,
  "Number of Items per Order": 1.1,
  "Average Markdown": 0.1,
  "Average Promotion/Discount": 0.2,
  "COGS Percentage": 0.5,
  "Churn Rate": 0.25,
},
  

  // ─── Marketing Expenses Defaults ────────────────────────────────────────────
  marketing: {
      "Email Cost per Click": 0,
  "Organic Search Cost per Click": 0,
  "Paid Search Cost per Click": 3.25,
  "Affiliates Cost per Click": 5,
  "Freight/Shipping per Order": 12,
  "Labor/Handling per Order": 5,
  "General Warehouse Rent": 150000,
  "Other": 100000,
  "Interest": 0,
  "Tax Rate": 0.25,
  "Direct Staff Hours per Year": 2000,
  "Direct Staff Number": 5,
  "Direct Staff Hourly Rate": 25,
  "Indirect Staff Hours per Year": 250000,
  "Indirect Staff Number": 2000,
  "Indirect Staff Hourly Rate": 2,
  "Part-Time Staff Hours per Year": 31,
  "Part-Time Staff Number": 124000,
  "Part-Time Staff Hourly Rate": 1000,
  "CEO Salary": 36000,
  "COO Salary": 10000,
  "CFO Salary": 5000,
  "Director of HR Salary": 5000,
  "CIO Salary": 5000,
  "Pension Cost per Staff": 1000,
  "Pension Total Cost": 15000,
  "Medical Insurance Cost per Staff": 1000,
  "Medical Insurance Total Cost": 15000,
  "Child Benefit Cost per Staff": 1000,
  "Child Benefit Total Cost": 15000,
  "Car Benefit Cost per Staff": 1000,
  "Car Benefit Total Cost": 15000,
  },
  

  // ─── Balance Sheet Defaults ─────────────────────────────────────────────────
  balance:{
       "Accounts Receivable Days": 0,
  "Inventory Days": 0,
  "Accounts Payable Days": 0,
  "Technology Development": 0,
  "Office Equipment": 0,
  "Technology Depreciation Years": 0,
  "Office Equipment Depreciation Years": 0,
  "Interest Rate (Default)": 0.03,
  "Equity Raised": 5000000,
  "Dividends Paid": 0,

  },
  
  // ─── Office Rent Breakdown Defaults ─────────────────────────────────────────
  // (OfficeRentBreakdownCard expects an array of { category, squareMeters, costPerSQM })
  officeRentRows: [
    {
      category: "Warehouse2",
      squareMeters: 100000,
      costPerSQM: 10
    },
    
  ],

  // ─── Professional Fees Defaults ──────────────────────────────────────────────
  // (ProfessionalFeesCard expects an array of { id, name, Cost })
  professionalFeesRows: [
    {  name: "Legal Cost", Cost: 100000 }
  ],

  // ─── Depreciation Breakdown Defaults ─────────────────────────────────────────
  // (DepreciationBreakdownCard expects an array of { id, name, amount, rate })
  depreciationBreakdown: [
    {
      id: 1,
      name: "Asset 1",
      amount: 1000000,
      rate: 10
    }
  ],

  // ─── Debt Issued Defaults ───────────────────────────────────────────────────
  // (DebtIssuedCard expects an array of { id, name, amount, interestRate, duration })
  debtIssued: [
    {
      id: 1,
      name: "Debt 1",
      amount: 10000,
      interestRate: 2,
      duration: 1
    }
  ]
};
