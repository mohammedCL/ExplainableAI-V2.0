import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
});

// A placeholder for the token. In a real app, this would come from a login process.
const getAuthToken = () => {
    // For now, we'll use a simple placeholder. You can manage this with context/state management later.
    return localStorage.getItem('authToken') || 'test_token';
};

apiClient.interceptors.request.use((config) => {
    const token = getAuthToken();
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// --- API Functions ---

export const uploadModelAndData = async (modelFile: File, dataFile: File, targetColumn: string) => {
    const formData = new FormData();
    formData.append('model_file', modelFile);
    formData.append('data_file', dataFile);
    formData.append('target_column', targetColumn);

    const response = await apiClient.post('/upload/model-and-data', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

export const getModelOverview = () => apiClient.get('/analysis/overview').then(res => res.data);
export const getClassificationStats = () => apiClient.get('/analysis/classification-stats').then(res => res.data);
export const getFeatureImportance = (method = 'shap') => apiClient.get(`/analysis/feature-importance?method=${method}`).then(res => res.data);
export const explainInstance = (instanceIdx: number) => apiClient.get(`/analysis/explain-instance/${instanceIdx}`).then(res => res.data);
export const listInstances = (sortBy: string = 'prediction', limit: number = 100) => apiClient.get(`/analysis/instances?sort_by=${sortBy}&limit=${limit}`).then(res => res.data);
export const performWhatIf = (features: Record<string, any>) => apiClient.post('/analysis/what-if', { features }).then(res => res.data);
export const getFeatureDependence = (featureName: string) => apiClient.get(`/analysis/feature-dependence/${featureName}`).then(res => res.data);
export const getFeatureInteractions = (feature1: string, feature2: string) => apiClient.get(`/analysis/feature-interactions?feature1=${feature1}&feature2=${feature2}`).then(res => res.data);
export const getDecisionTree = () => apiClient.get('/analysis/decision-tree').then(res => res.data);

// Enterprise feature APIs
export const getFeaturesMetadata = () => apiClient.get('/api/features').then(res => res.data);
export const postCorrelation = (features: string[]) => apiClient.post('/api/correlation', { features }).then(res => res.data);
export const postAdvancedImportance = (body: { method: string; sort_by: string; top_n: number; visualization: string; }) => apiClient.post('/api/feature-importance', body).then(res => res.data);

// Section 2 - Classification
export const postRocAnalysis = () => apiClient.get('/api/roc-analysis').then(res => res.data);
export const postThresholdAnalysis = (num_thresholds = 50) => apiClient.post(`/api/threshold-analysis?num_thresholds=${num_thresholds}`).then(res => res.data);

// Section 3 - Individual prediction summary
export const postIndividualPrediction = (instance_idx: number) => apiClient.post('/api/individual-prediction', { instance_idx }).then(res => res.data);

// Section 4 - Dependence
export const postPartialDependence = (feature: string, num_points = 20) => apiClient.post('/api/partial-dependence', { feature, num_points }).then(res => res.data);
export const postShapDependence = (feature: string, color_by?: string) => apiClient.post('/api/shap-dependence', { feature, color_by }).then(res => res.data);
export const postIcePlot = (feature: string, num_points = 20, num_instances = 20) => apiClient.post('/api/ice-plot', { feature, num_points, num_instances }).then(res => res.data);

// Section 5 - Interactions
export const postInteractionNetwork = (top_k = 30, sample_rows = 200) => apiClient.post('/api/interaction-network', { top_k, sample_rows }).then(res => res.data);
export const postPairwiseAnalysis = (feature1: string, feature2: string, color_by?: string, sample_size = 1000) => apiClient.post('/api/pairwise-analysis', { feature1, feature2, color_by, sample_size }).then(res => res.data);

// AI Explanation API
export const explainWithAI = (analysisType: string, analysisData: any) => apiClient.post('/analysis/explain-with-ai', { analysis_type: analysisType, analysis_data: analysisData }).then(res => res.data);