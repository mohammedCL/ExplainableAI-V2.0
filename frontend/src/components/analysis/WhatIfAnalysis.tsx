import React, { useState } from 'react';
import { AlertCircle, Sliders, Zap, BarChart3, RefreshCw, Save } from 'lucide-react';

const FeatureSlider = ({ name, value, min, max, step, onChange }: {
    name: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (value: number) => void;
}) => (
    <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
        <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium">{name.replace('_', ' ')}</label>
            <input
                type="number"
                value={value}
                onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
                className="w-20 px-2 py-1 text-xs border rounded bg-white dark:bg-gray-600"
                step={step}
                min={min}
                max={max}
            />
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{min.toLocaleString()}</span>
            <span>{max.toLocaleString()}</span>
        </div>
    </div>
);

const PredictionDisplay = ({ prediction, confidence }: { prediction: number; confidence: number }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Zap className="mr-2 text-blue-600" />
            Current Prediction
        </h3>
        <div className="text-center">
            <div className="inline-block p-6 rounded-full bg-green-50 dark:bg-green-900/20 mb-4">
                <div className="text-4xl font-bold text-green-600">
                    {(prediction * 100).toFixed(1)}%
                </div>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Approval Probability
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
            Impact Analysis
        </h3>
        <div className="space-y-3">
            {impacts.map((feature, index) => (
                <div key={index} className="flex items-center justify-between">
                    <span className="text-sm font-medium w-32 truncate">{feature.name.replace('_', ' ')}</span>
                    <div className="flex-1 mx-4">
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                            <div
                                className="h-2 rounded-full bg-blue-500"
                                style={{ width: `${Math.abs(feature.impact) * 100}%` }}
                            ></div>
                        </div>
                    </div>
                    <span className="text-xs font-mono w-12 text-right">
                        {feature.impact.toFixed(1)}
                    </span>
                </div>
            ))}
        </div>
    </div>
);

const WhatIfAnalysis: React.FC<{ modelType?: string }> = () => {
    const [features, setFeatures] = useState({
        Customer_Age: 35,
        Annual_Income: 75000,
        Credit_Score: 720,
        Employment_Type: 'Full-time',
        Education_Level: 'Bachelor'
    });

    const [prediction, setPrediction] = useState(0.95);
    const [confidence, setConfidence] = useState(0.88);
    const [error, setError] = useState('');

    const featureImpacts = [
        { name: 'Customer_Age', impact: 0.1 },
        { name: 'Annual_Income', impact: 0.2 },
        { name: 'Credit_Score', impact: 0.3 }
    ];

    const updateFeature = (name: string, value: number) => {
        setFeatures(prev => ({ ...prev, [name]: value }));
        // Simulate prediction update based on feature changes
        let newPrediction = 0.5;
        if (name === 'Credit_Score') {
            newPrediction = Math.min(0.99, Math.max(0.01, value / 850));
        } else if (name === 'Annual_Income') {
            newPrediction = Math.min(0.99, Math.max(0.01, value / 150000));
        } else if (name === 'Customer_Age') {
            newPrediction = Math.min(0.99, Math.max(0.01, (80 - Math.abs(value - 40)) / 80));
        }
        setPrediction(newPrediction);
        setConfidence(Math.min(0.99, Math.max(0.60, newPrediction + (Math.random() - 0.5) * 0.1)));
    };

    const resetToDefault = () => {
        setFeatures({
            Customer_Age: 35,
            Annual_Income: 75000,
            Credit_Score: 720,
            Employment_Type: 'Full-time',
            Education_Level: 'Bachelor'
        });
        setPrediction(0.95);
        setConfidence(0.88);
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="p-6 max-w-full">
                <div className="space-y-6">
                    <div className="flex items-center justify-between">
                        <h1 className="text-3xl font-bold">What-If Analysis</h1>
                        <p className="text-sm text-gray-500">Interactive scenario analysis to explore how changes in features affect predictions</p>
                    </div>

                    <div className="flex space-x-2">
                        <button
                            onClick={resetToDefault}
                            className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 flex items-center space-x-2"
                        >
                            <RefreshCw className="w-4 h-4" />
                            <span>Reset</span>
                        </button>
                        <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center space-x-2">
                            <Save className="w-4 h-4" />
                            <span>Save Scenario</span>
                        </button>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Feature Controls */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                            <h2 className="text-lg font-semibold mb-4 flex items-center">
                                <Sliders className="mr-2" />
                                Feature Controls
                            </h2>

                            <FeatureSlider
                                name="Customer_Age"
                                value={features.Customer_Age}
                                min={18}
                                max={80}
                                step={1}
                                onChange={(value) => updateFeature('Customer_Age', value)}
                            />

                            <FeatureSlider
                                name="Annual_Income"
                                value={features.Annual_Income}
                                min={20000}
                                max={200000}
                                step={1000}
                                onChange={(value) => updateFeature('Annual_Income', value)}
                            />

                            <FeatureSlider
                                name="Credit_Score"
                                value={features.Credit_Score}
                                min={300}
                                max={850}
                                step={10}
                                onChange={(value) => updateFeature('Credit_Score', value)}
                            />

                            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                <label className="block text-sm font-medium mb-2">Employment Type</label>
                                <select
                                    value={features.Employment_Type}
                                    onChange={(e) => setFeatures(prev => ({ ...prev, Employment_Type: e.target.value }))}
                                    className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-600"
                                >
                                    <option value="Full-time">Full-time</option>
                                    <option value="Part-time">Part-time</option>
                                    <option value="Self-employed">Self-employed</option>
                                    <option value="Unemployed">Unemployed</option>
                                </select>
                            </div>

                            <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                <label className="block text-sm font-medium mb-2">Education Level</label>
                                <select
                                    value={features.Education_Level}
                                    onChange={(e) => setFeatures(prev => ({ ...prev, Education_Level: e.target.value }))}
                                    className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-600"
                                >
                                    <option value="High School">High School</option>
                                    <option value="Bachelor">Bachelor</option>
                                    <option value="Master">Master</option>
                                    <option value="PhD">PhD</option>
                                </select>
                            </div>
                        </div>

                        {/* Current Prediction */}
                        <PredictionDisplay prediction={prediction} confidence={confidence} />
                    </div>

                    {/* Feature Impact Analysis */}
                    <FeatureImpactChart impacts={featureImpacts} />

                    {error && (
                        <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                            <AlertCircle className="mr-2 flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default WhatIfAnalysis;
