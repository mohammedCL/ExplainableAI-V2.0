import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import ModelOverview from './components/analysis/ModelOverview';
import ClassificationStats from './components/analysis/ClassificationStats';
import FeatureImportance from './components/analysis/FeatureImportance';
import IndividualPredictions from './components/analysis/IndividualPredictions';
import WhatIfAnalysis from './components/analysis/WhatIfAnalysis';
import FeatureDependence from './components/analysis/FeatureDependence';
import FeatureInteractions from './components/analysis/FeatureInteractions';
import UploadPage from './components/analysis/UploadPage';
import DecisionTreesWrapper from './components/analysis/DecisionTreesWrapper';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/upload" element={<UploadPage />} /> {/* New upload route */}
        <Route path="/" element={<MainLayout />}>
          {/* Default route redirects to model overview */}
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<ModelOverview modelType="classification" />} />
          <Route path="classification-stats" element={<ClassificationStats />} />
          <Route path="feature-importance" element={<FeatureImportance modelType="classification" />} />
          <Route path="individual-predictions" element={<IndividualPredictions modelType="classification" />} />
          <Route path="what-if" element={<WhatIfAnalysis modelType="classification" />} />
          <Route path="feature-dependence" element={<FeatureDependence modelType="classification" />} />
          <Route path="feature-interactions" element={<FeatureInteractions modelType="classification" />} />
          <Route path="decision-trees" element={<DecisionTreesWrapper />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;