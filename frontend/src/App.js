
import React from 'react';
import { Route,Routes} from 'react-router-dom';
import FinanceDashboard from './pages/FinanceDashboard';

function App() {
  return (
     <Routes>
      <Route path="/ecommerce" element={<FinanceDashboard />} />
   </Routes>
  )
   
   
}

export default App;
