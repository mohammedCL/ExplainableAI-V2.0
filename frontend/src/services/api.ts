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
export const performWhatIf = (features: Record<string, any>) => apiClient.post('/analysis/what-if', { features }).then(res => res.data);
export const getFeatureDependence = (featureName: string) => apiClient.get(`/analysis/feature-dependence/${featureName}`).then(res => res.data);
export const getFeatureInteractions = (feature1: string, feature2: string) => apiClient.get(`/analysis/feature-interactions?feature1=${feature1}&feature2=${feature2}`).then(res => res.data);
export const getDecisionTree = () => apiClient.get('/analysis/decision-tree').then(res => res.data);