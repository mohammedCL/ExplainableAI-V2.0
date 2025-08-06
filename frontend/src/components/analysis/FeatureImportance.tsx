import React, { useState, useEffect } from 'react';
import { getFeatureImportance } from '../../services/api';
import { AlertCircle, Loader2, BarChart3, ArrowUp, ArrowDown, Info } from 'lucide-react';

const FeatureBar = ({ name, importance, maxImportance }: {
    name: string;
    importance: number;
    maxImportance: number;
}) => {
    const widthPercentage = Math.abs(importance / maxImportance) * 100;
    const isPositive = importance > 0;

    return (
        <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-sm truncate max-w-xs" title={name}>
                    {name}
                </span>
                <div className="flex items-center space-x-2">
                    {isPositive ? (
                        <ArrowUp className="w-4 h-4 text-green-500" />
                    ) : (
                        <ArrowDown className="w-4 h-4 text-red-500" />
                    )}
                    <span className="text-sm font-mono">
                        {importance.toFixed(4)}
                    </span>
                </div>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                <div
                    className={`h-2 rounded-full ${isPositive ? 'bg-green-500' : 'bg-red-500'
                        }`}
                    style={{ width: `${Math.max(widthPercentage, 5)}%` }}
                ></div>
            </div>
        </div>
    );
};

const MethodCard = ({ method, isActive, onClick, description }: {
    method: string;
    isActive: boolean;
    onClick: () => void;
    description: string;
}) => (
    <div
        className={`p-4 rounded-lg cursor-pointer border-2 transition-all ${isActive
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
            }`}
        onClick={onClick}
    >
        <div className="font-medium text-sm">{method.toUpperCase()}</div>
        <div className="text-xs text-gray-500 mt-1">{description}</div>
    </div>
);

const FeatureImportance: React.FC<{ modelType?: string }> = () => {
    const [importance, setImportance] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [method, setMethod] = useState('shap');

    useEffect(() => {
        const fetchImportance = async () => {
            try {
                setLoading(true);
                const data = await getFeatureImportance(method);
                setImportance(data);
            } catch (err: any) {
                setError(err.response?.data?.detail || 'Failed to fetch feature importance.');
            } finally {
                setLoading(false);
            }
        };
        fetchImportance();
    }, [method]);

    if (loading) {
        return <div className="p-6 flex justify-center items-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>;
    }
    if (error) {
        return <div className="p-6 text-red-500"><AlertCircle className="inline-block mr-2" />{error}</div>;
    }
    if (!importance) {
        return <div className="p-6">No feature importance data available.</div>;
    }

    const features = importance.features || [];
    const maxImportance = Math.max(...features.map((f: any) => Math.abs(f.importance)));

    const methodDescriptions = {
        shap: "SHAP values show feature contributions to individual predictions",
        permutation: "Measures importance by feature shuffling impact on accuracy",
        tree: "Tree-based importance from model's internal structure"
    };

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold flex items-center">
                    <BarChart3 className="mr-3 text-blue-600" />
                    Feature Importance Analysis
                </h1>
                <div className="flex items-center space-x-2 text-sm text-gray-500">
                    <Info className="w-4 h-4" />
                    <span>Method: {method.toUpperCase()}</span>
                </div>
            </div>

            {/* Method Selection */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold mb-4">Analysis Method</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Object.entries(methodDescriptions).map(([methodName, description]) => (
                        <MethodCard
                            key={methodName}
                            method={methodName}
                            isActive={method === methodName}
                            onClick={() => setMethod(methodName)}
                            description={description}
                        />
                    ))}
                </div>
            </div>

            {/* Feature Importance Results */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-semibold">Feature Rankings</h2>
                    <div className="text-sm text-gray-500">
                        {features.length} features analyzed
                    </div>
                </div>

                {features.length > 0 ? (
                    <div className="space-y-2">
                        {features
                            .sort((a: any, b: any) => Math.abs(b.importance) - Math.abs(a.importance))
                            .slice(0, 15) // Show top 15 features
                            .map((feature: any, index: number) => (
                                <div key={index} className="flex items-center space-x-4">
                                    <div className="w-6 text-sm font-medium text-gray-400">
                                        #{index + 1}
                                    </div>
                                    <div className="flex-1">
                                        <FeatureBar
                                            name={feature.name}
                                            importance={feature.importance}
                                            maxImportance={maxImportance}
                                        />
                                    </div>
                                </div>
                            ))}
                    </div>
                ) : (
                    <div className="text-center py-8 text-gray-500">
                        No feature importance data available
                    </div>
                )}
            </div>

            {/* Summary Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
                    <div className="text-2xl font-bold text-blue-600">
                        {features.length}
                    </div>
                    <div className="text-sm text-gray-500">Total Features</div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
                    <div className="text-2xl font-bold text-green-600">
                        {features.filter((f: any) => f.importance > 0).length}
                    </div>
                    <div className="text-sm text-gray-500">Positive Impact</div>
                </div>
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 text-center">
                    <div className="text-2xl font-bold text-red-600">
                        {features.filter((f: any) => f.importance < 0).length}
                    </div>
                    <div className="text-sm text-gray-500">Negative Impact</div>
                </div>
            </div>
        </div>
    );
};

export default FeatureImportance;
