import React, { useState, useEffect } from 'react';
import { AlertCircle, Sliders, Zap, BarChart3, RefreshCw, Loader2, Target } from 'lucide-react';
import { performWhatIf, getModelOverview } from '../../services/api';

const FeatureSlider = ({ name, value, min, max, step, onChange }: {
    name: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (value: number) => void;
}) => (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium">{name.replace('_', ' ')}</label>
            <span className="text-sm font-mono text-blue-600">{value}</span>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{min}</span>
            <span>{max}</span>
        </div>
    </div>
);

const FeatureInput = ({ name, value, type, onChange }: {
    name: string;
    value: any;
    type: 'text' | 'number';
    onChange: (value: any) => void;
}) => (
    <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <label className="block text-sm font-medium mb-2">{name.replace('_', ' ')}</label>
        <input
            type={type}
            value={value}
            onChange={(e) => onChange(type === 'number' ? Number(e.target.value) : e.target.value)}
            className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-600 dark:border-gray-500"
        />
    </div>
);

const PredictionDisplay = ({ prediction, confidence }: { prediction: number; confidence: number }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Target className="mr-2 text-green-600" />
            Current Prediction
        </h3>
        <div className="text-center">
            <div className="text-4xl font-bold text-green-600 mb-2">
                {(prediction * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Prediction Probability
            </div>
            <div className="text-xs text-gray-500">
                Confidence: {(confidence * 100).toFixed(1)}%
            </div>
        </div>
    </div>
);

const FeatureImpactChart = ({ impacts }: { impacts: Array<{ name: string, impact: number }> }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="mr-2 text-purple-600" />
            Feature Impact Analysis
        </h3>
        <div className="space-y-3">
            {impacts.map((feature, index) => (
                <div key={index} className="flex items-center justify-between">
                    <span className="text-sm font-medium w-32 truncate">{feature.name.replace('_', ' ')}</span>
                    <div className="flex-1 mx-4">
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div
                                className={`h-2 rounded-full ${feature.impact >= 0 ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ width: `${Math.abs(feature.impact) * 100}%` }}
                            ></div>
                        </div>
                    </div>
                    <span className="text-xs font-mono w-12 text-right">
                        {feature.impact.toFixed(3)}
                    </span>
                </div>
            ))}
        </div>
    </div>
);

const WhatIfAnalysis: React.FC<{ modelType?: string }> = () => {
    const [features, setFeatures] = useState<Record<string, any>>({});
    const [prediction, setPrediction] = useState<number>(0);
    const [confidence, setConfidence] = useState<number>(0);
    const [featureImpacts, setFeatureImpacts] = useState<Array<{ name: string, impact: number }>>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [featureList, setFeatureList] = useState<string[]>([]);
    const [modelLoaded, setModelLoaded] = useState(false);

    // Load model overview and feature names on component mount
    useEffect(() => {
        const loadModelInfo = async () => {
            try {
                const overview = await getModelOverview();
                const featureNames = overview.feature_names || [];
                setFeatureList(featureNames);
                setModelLoaded(true);
                
                // Initialize features with default values based on data types
                const initialFeatures: Record<string, any> = {};
                featureNames.forEach((feature: string) => {
                    // Use reasonable defaults - in a real app, these could come from feature metadata
                    if (feature.toLowerCase().includes('age')) {
                        initialFeatures[feature] = 35;
                    } else if (feature.toLowerCase().includes('income') || feature.toLowerCase().includes('salary')) {
                        initialFeatures[feature] = 50000;
                    } else if (feature.toLowerCase().includes('score') || feature.toLowerCase().includes('credit')) {
                        initialFeatures[feature] = 700;
                    } else {
                        initialFeatures[feature] = 0; // Default numeric value
                    }
                });
                setFeatures(initialFeatures);
                
                // Get initial prediction
                await performPrediction(initialFeatures);
                
            } catch (err: any) {
                setError('Failed to load model information. Please ensure a model is uploaded.');
                setModelLoaded(false);
            }
        };
        
        loadModelInfo();
    }, []);

    const performPrediction = async (currentFeatures: Record<string, any>) => {
        if (!modelLoaded) return;
        
        setLoading(true);
        setError('');
        
        try {
            const result = await performWhatIf(currentFeatures);
            setPrediction(result.prediction || 0);
            setConfidence(0.8); // Mock confidence since backend doesn't return it
            
            // Calculate mock feature impacts (in a real app, this would come from SHAP or similar)
            const impacts = Object.keys(currentFeatures).map(feature => ({
                name: feature,
                impact: Math.random() * 0.4 - 0.2 // Random impact between -0.2 and 0.2
            }));
            setFeatureImpacts(impacts);
            
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to get prediction');
            setPrediction(0);
            setConfidence(0);
        } finally {
            setLoading(false);
        }
    };

    const handleFeatureChange = async (featureName: string, value: any) => {
        const newFeatures = { ...features, [featureName]: value };
        setFeatures(newFeatures);
        
        // Debounce prediction calls
        clearTimeout((window as any).predictionTimeout);
        (window as any).predictionTimeout = setTimeout(() => {
            performPrediction(newFeatures);
        }, 500);
    };

    if (!modelLoaded && !error) {
        return (
            <div className="p-6 flex justify-center items-center h-full">
                <Loader2 className="w-8 h-8 animate-spin" />
                <span className="ml-2">Loading model information...</span>
            </div>
        );
    }

    if (error && !modelLoaded) {
        return (
            <div className="p-6 text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-red-600 mb-2">Model Not Available</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">{error}</p>
                <p className="text-sm text-gray-500">Please upload a model and dataset first.</p>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold flex items-center">
                    <Sliders className="mr-3 text-blue-600" />
                    What-If Analysis
                </h1>
                <div className="flex items-center space-x-2">
                    <button 
                        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
                        onClick={() => performPrediction(features)}
                        disabled={loading}
                    >
                        {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <RefreshCw className="w-4 h-4 mr-2" />}
                        {loading ? 'Updating...' : 'Refresh'}
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Feature Controls */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-xl font-semibold mb-4 flex items-center">
                            <Zap className="mr-2 text-orange-600" />
                            Adjust Feature Values
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {featureList.map((feature) => {
                                const value = features[feature] || 0;
                                const isNumeric = typeof value === 'number';
                                
                                if (isNumeric) {
                                    // Determine appropriate range based on feature name
                                    let min = 0, max = 100, step = 1;
                                    if (feature.toLowerCase().includes('age')) {
                                        min = 18; max = 80; step = 1;
                                    } else if (feature.toLowerCase().includes('income')) {
                                        min = 20000; max = 200000; step = 1000;
                                    } else if (feature.toLowerCase().includes('score')) {
                                        min = 300; max = 850; step = 10;
                                    }
                                    
                                    return (
                                        <FeatureSlider
                                            key={feature}
                                            name={feature}
                                            value={value}
                                            min={min}
                                            max={max}
                                            step={step}
                                            onChange={(newValue) => handleFeatureChange(feature, newValue)}
                                        />
                                    );
                                } else {
                                    return (
                                        <FeatureInput
                                            key={feature}
                                            name={feature}
                                            value={value}
                                            type="text"
                                            onChange={(newValue) => handleFeatureChange(feature, newValue)}
                                        />
                                    );
                                }
                            })}
                        </div>
                    </div>

                    {/* Feature Impact Analysis */}
                    {featureImpacts.length > 0 && (
                        <FeatureImpactChart impacts={featureImpacts} />
                    )}

                    {error && (
                        <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                            <AlertCircle className="mr-2 flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}
                </div>

                {/* Prediction Display */}
                <div>
                    <PredictionDisplay prediction={prediction} confidence={confidence} />
                </div>
            </div>
        </div>
    );
};

export default WhatIfAnalysis;
