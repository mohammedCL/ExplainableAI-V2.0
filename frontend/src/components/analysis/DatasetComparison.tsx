import React, { useState, useEffect } from 'react';
import { getDatasetComparison } from '../../services/api';
import { AlertCircle, Loader2, Database, TrendingUp, TrendingDown } from 'lucide-react';

const MetricCard = ({ 
    title, 
    trainValue, 
    testValue, 
    format = 'number',
    icon 
}: { 
    title: string; 
    trainValue: number; 
    testValue: number; 
    format?: 'percentage' | 'number';
    icon: React.ReactNode;
}) => {
    const formatValue = (value: number) => {
        if (format === 'percentage') return `${value.toFixed(1)}%`;
        if (Number.isInteger(value)) return value.toString();
        return value.toFixed(3);
    };

    const difference = testValue - trainValue;
    const isHigher = difference > 0;
    const significantDiff = Math.abs(difference) > 0.05; // 5% threshold

    return (
        <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="flex items-center mb-3">
                <div className="p-2 bg-blue-100 dark:bg-blue-900/50 rounded-full">
                    {icon}
                </div>
                <h3 className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">{title}</h3>
            </div>
            
            <div className="space-y-2">
                <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-500">Training:</span>
                    <span className="font-semibold text-blue-600">{formatValue(trainValue)}</span>
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-500">Testing:</span>
                    <span className="font-semibold text-green-600">{formatValue(testValue)}</span>
                </div>
                
                {significantDiff && (
                    <div className={`flex items-center justify-center text-xs px-2 py-1 rounded ${
                        isHigher 
                            ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-200'
                            : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200'
                    }`}>
                        {isHigher ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                        {formatValue(Math.abs(difference))} {isHigher ? 'higher' : 'lower'}
                    </div>
                )}
            </div>
        </div>
    );
};

const DatasetComparison: React.FC = () => {
    const [comparison, setComparison] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchComparison = async () => {
            try {
                setLoading(true);
                const data = await getDatasetComparison();
                setComparison(data);
            } catch (err: any) {
                setError(err.response?.data?.detail || 'Failed to fetch dataset comparison.');
            } finally {
                setLoading(false);
            }
        };
        fetchComparison();
    }, []);

    if (loading) {
        return (
            <div className="p-6 flex justify-center items-center h-64">
                <Loader2 className="w-8 h-8 animate-spin" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <AlertCircle className="inline-block mr-2" />
                {error}
            </div>
        );
    }

    if (!comparison) {
        return (
            <div className="p-6 text-gray-500 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <Database className="inline-block mr-2" />
                No dataset comparison available. This feature is only available when using separate train/test datasets.
            </div>
        );
    }

    const { train_metrics, test_metrics, overfitting_metrics } = comparison;

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-900">
            <div className="flex items-center space-x-3">
                <Database className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Dataset Comparison</h2>
            </div>

            {/* Performance Metrics Comparison */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Performance Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricCard
                        title="Accuracy"
                        trainValue={train_metrics.accuracy * 100}
                        testValue={test_metrics.accuracy * 100}
                        format="percentage"
                        icon={<Database className="w-4 h-4 text-blue-600" />}
                    />
                    <MetricCard
                        title="Precision"
                        trainValue={train_metrics.precision * 100}
                        testValue={test_metrics.precision * 100}
                        format="percentage"
                        icon={<Database className="w-4 h-4 text-blue-600" />}
                    />
                    <MetricCard
                        title="Recall"
                        trainValue={train_metrics.recall * 100}
                        testValue={test_metrics.recall * 100}
                        format="percentage"
                        icon={<Database className="w-4 h-4 text-blue-600" />}
                    />
                    <MetricCard
                        title="F1 Score"
                        trainValue={train_metrics.f1_score * 100}
                        testValue={test_metrics.f1_score * 100}
                        format="percentage"
                        icon={<Database className="w-4 h-4 text-blue-600" />}
                    />
                </div>
            </div>

            {/* Overfitting Analysis */}
            {overfitting_metrics && (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Overfitting Analysis</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="text-center">
                            <div className={`text-2xl font-bold mb-2 ${
                                overfitting_metrics.overfitting_score < 0.1 
                                    ? 'text-green-600' 
                                    : overfitting_metrics.overfitting_score < 0.2 
                                        ? 'text-yellow-600' 
                                        : 'text-red-600'
                            }`}>
                                {(overfitting_metrics.overfitting_score * 100).toFixed(1)}%
                            </div>
                            <div className="text-sm text-gray-500">Overfitting Score</div>
                        </div>
                        <div className="text-center">
                            <div className={`text-2xl font-bold mb-2 ${
                                overfitting_metrics.is_overfitting ? 'text-red-600' : 'text-green-600'
                            }`}>
                                {overfitting_metrics.is_overfitting ? 'Yes' : 'No'}
                            </div>
                            <div className="text-sm text-gray-500">Overfitting Detected</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold mb-2 text-blue-600">
                                {overfitting_metrics.generalization_gap.toFixed(3)}
                            </div>
                            <div className="text-sm text-gray-500">Generalization Gap</div>
                        </div>
                    </div>
                    
                    {overfitting_metrics.is_overfitting && (
                        <div className="mt-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                            <p className="text-sm text-yellow-700 dark:text-yellow-200">
                                <strong>Warning:</strong> Potential overfitting detected. The model performs significantly better on training data than test data.
                                Consider using regularization techniques or collecting more training data.
                            </p>
                        </div>
                    )}
                </div>
            )}

            {/* Dataset Statistics */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Dataset Statistics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Training Dataset</h4>
                        <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                                <span>Samples:</span>
                                <span className="font-semibold">{comparison.train_samples?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Missing Data:</span>
                                <span className="font-semibold">{comparison.train_missing_pct?.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Duplicates:</span>
                                <span className="font-semibold">{comparison.train_duplicates_pct?.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <h4 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Testing Dataset</h4>
                        <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                                <span>Samples:</span>
                                <span className="font-semibold">{comparison.test_samples?.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Missing Data:</span>
                                <span className="font-semibold">{comparison.test_missing_pct?.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span>Duplicates:</span>
                                <span className="font-semibold">{comparison.test_duplicates_pct?.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DatasetComparison;
