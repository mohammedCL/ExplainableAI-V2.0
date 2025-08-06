import React, { useState, useEffect } from 'react';
import { getModelOverview } from '../../services/api';
import { Target, BarChart3, Binary, AlertCircle, Info, Loader2 } from 'lucide-react';

const MetricCard = ({ title, value, format, icon }: { title: string; value: number; format: 'percentage' | 'number'; icon: React.ReactNode }) => (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
        <div className="flex items-center">
            <div className="p-3 bg-blue-100 dark:bg-blue-900/50 rounded-full">{icon}</div>
            <div className="ml-4">
                <p className="text-sm text-gray-500 dark:text-gray-400">{title}</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {format === 'percentage' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
                </p>
            </div>
        </div>
    </div>
);

const ModelOverview: React.FC<{ modelType: string }> = ({ modelType }) => {
    const [overview, setOverview] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchOverview = async () => {
            try {
                setLoading(true);
                const data = await getModelOverview();
                setOverview(data);
            } catch (err: any) {
                setError(err.response?.data?.detail || 'Failed to fetch model overview.');
            } finally {
                setLoading(false);
            }
        };
        fetchOverview();
    }, []);

    if (loading) {
        return <div className="p-6 flex justify-center items-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>;
    }
    if (error) {
        return <div className="p-6 text-red-500"><AlertCircle className="inline-block mr-2" />{error}</div>;
    }
    if (!overview) {
        return <div className="p-6">No overview data available.</div>;
    }

    const { performance_metrics: metrics } = overview;

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <h1 className="text-3xl font-bold">Model Overview</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                <MetricCard title="Accuracy" value={metrics.accuracy} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="Precision" value={metrics.precision} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="Recall" value={metrics.recall} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="F1 Score" value={metrics.f1_score} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="AUC" value={metrics.auc} format="number" icon={<Target className="w-6 h-6 text-blue-600" />} />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><Info className="mr-2" /> Model Information</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div><strong>Model ID:</strong> <span className="text-gray-600 dark:text-gray-400">{overview.model_id}</span></div>
                    <div><strong>Name:</strong> <span className="text-gray-600 dark:text-gray-400">{overview.name}</span></div>
                    <div><strong>Type:</strong> <span className="text-gray-600 dark:text-gray-400">{overview.model_type}</span></div>
                    <div><strong>Version:</strong> <span className="text-gray-600 dark:text-gray-400">{overview.version}</span></div>
                    <div><strong>Framework:</strong> <span className="text-gray-600 dark:text-gray-400">{overview.framework}</span></div>
                    <div><strong>Status:</strong> <span className="text-green-500 font-semibold">{overview.status}</span></div>
                </div>
            </div>
        </div>
    );
};

export default ModelOverview;