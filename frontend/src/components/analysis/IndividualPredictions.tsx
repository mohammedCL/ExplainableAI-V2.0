import React, { useState } from 'react';
import { explainInstance } from '../../services/api';
import { AlertCircle, Loader2, User, Target, TrendingUp, TrendingDown, Info } from 'lucide-react';

const FeatureContribution = ({ name, value, contribution, maxContribution }: {
    name: string;
    value: any;
    contribution: number;
    maxContribution: number;
}) => {
    const widthPercentage = Math.abs(contribution / maxContribution) * 100;
    const isPositive = contribution > 0;

    return (
        <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg mb-3">
            <div className="flex justify-between items-center mb-2">
                <div className="flex-1">
                    <span className="font-medium text-sm">{name}</span>
                    <span className="text-xs text-gray-500 ml-2">= {value}</span>
                </div>
                <div className="flex items-center space-x-2">
                    {isPositive ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                    ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm font-mono">
                        {contribution > 0 ? '+' : ''}{contribution.toFixed(3)}
                    </span>
                </div>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-1.5">
                <div
                    className={`h-1.5 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'
                        }`}
                    style={{ width: `${Math.max(widthPercentage, 3)}%` }}
                ></div>
            </div>
        </div>
    );
};

const PredictionCard = ({ prediction, confidence }: {
    prediction: any;
    confidence: number;
}) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center">
                <Target className="mr-2 text-blue-600" />
                Prediction Result
            </h3>
            <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">
                    {typeof prediction === 'number' ? prediction.toFixed(3) : String(prediction)}
                </div>
                <div className="text-sm text-gray-500">Predicted Value</div>
            </div>
        </div>
        <div className="mt-4">
            <div className="flex justify-between text-sm mb-2">
                <span>Confidence</span>
                <span>{(confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div
                    className="h-2 bg-blue-500 rounded-full transition-all duration-300"
                    style={{ width: `${confidence * 100}%` }}
                ></div>
            </div>
        </div>
    </div>
);

const IndividualPredictions: React.FC<{ modelType?: string }> = () => {
    const [explanation, setExplanation] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [instanceIdx, setInstanceIdx] = useState(0);

    const fetchExplanation = async () => {
        try {
            setLoading(true);
            setError('');
            const data = await explainInstance(instanceIdx);
            setExplanation(data);
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Failed to fetch instance explanation.');
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        fetchExplanation();
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="p-6 max-w-full">
                <div className="space-y-6">
                    <div className="flex items-center justify-between">
                        <h1 className="text-3xl font-bold flex items-center">
                            <User className="mr-3 text-blue-600" />
                            Individual Prediction Analysis
                        </h1>
                        <div className="flex items-center space-x-2 text-sm text-gray-500">
                            <Info className="w-4 h-4" />
                            <span>Explain specific instances</span>
                        </div>
                    </div>

                    {/* Input Form */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold mb-4">Select Instance to Analyze</h2>
                        <form onSubmit={handleSubmit} className="flex items-end space-x-4">
                            <div className="flex-1">
                                <label className="block text-sm font-medium mb-2">Instance Index:</label>
                                <input
                                    type="number"
                                    value={instanceIdx}
                                    onChange={(e) => setInstanceIdx(parseInt(e.target.value) || 0)}
                                    className="w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
                                    min="0"
                                    placeholder="Enter row number from dataset"
                                />
                            </div>
                            <button
                                type="submit"
                                disabled={loading}
                                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center space-x-2"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        <span>Analyzing...</span>
                                    </>
                                ) : (
                                    <>
                                        <Target className="w-4 h-4" />
                                        <span>Explain Instance</span>
                                    </>
                                )}
                            </button>
                        </form>
                        <p className="text-xs text-gray-500 mt-2">
                            Enter the index of the data point you want to analyze (0-based indexing)
                        </p>
                    </div>

                    {error && (
                        <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                            <AlertCircle className="mr-2 flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {explanation && !loading && (
                        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                            {/* Left Column - Prediction Result and Summary */}
                            <div className="xl:col-span-1 space-y-6">
                                <PredictionCard
                                    prediction={explanation.prediction}
                                    confidence={explanation.prediction_probability || 0.5}
                                />

                                {/* Summary Statistics */}
                                <div className="grid grid-cols-1 gap-4">
                                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
                                        <div className="text-lg font-bold text-green-600">
                                            {explanation.shap_values ?
                                                explanation.shap_values.filter((v: number) => v > 0).length : 0}
                                        </div>
                                        <div className="text-sm text-gray-500">Positive Contributors</div>
                                    </div>
                                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
                                        <div className="text-lg font-bold text-red-600">
                                            {explanation.shap_values ?
                                                explanation.shap_values.filter((v: number) => v < 0).length : 0}
                                        </div>
                                        <div className="text-sm text-gray-500">Negative Contributors</div>
                                    </div>
                                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 text-center">
                                        <div className="text-lg font-bold text-blue-600">
                                            {explanation.base_value ? explanation.base_value.toFixed(3) : 'N/A'}
                                        </div>
                                        <div className="text-sm text-gray-500">Base Value</div>
                                    </div>
                                </div>
                            </div>

                            {/* Right Column - Feature Contributions */}
                            <div className="xl:col-span-2">
                                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 h-full">
                                    <h3 className="text-lg font-semibold mb-4">Feature Contributions</h3>
                                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                                        How each feature contributed to this specific prediction
                                    </p>

                                    {explanation.shap_values && explanation.feature_names ? (
                                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                            {explanation.feature_names.map((name: string, index: number) => {
                                                const contribution = explanation.shap_values[index];
                                                const featureValue = explanation.feature_values?.[index] || 'N/A';
                                                const maxContribution = Math.max(...explanation.shap_values.map((v: number) => Math.abs(v)));

                                                return (
                                                    <FeatureContribution
                                                        key={index}
                                                        name={name}
                                                        value={featureValue}
                                                        contribution={contribution}
                                                        maxContribution={maxContribution}
                                                    />
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="text-center py-8 text-gray-500">
                                            No feature contribution data available
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {!explanation && !loading && !error && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-8 text-center">
                            <User className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                            <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">
                                No Analysis Yet
                            </h3>
                            <p className="text-sm text-gray-500">
                                Enter an instance index above to get a detailed explanation of the model's prediction.
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default IndividualPredictions;
