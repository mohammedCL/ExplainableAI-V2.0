import React, { useState, useEffect } from 'react';
import { getModelOverview } from '../../services/api';
import { Target, AlertCircle, Info, Loader2 } from 'lucide-react';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

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

const ModelOverview: React.FC<{ modelType: string }> = () => {
    const [overview, setOverview] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [showAIExplanation, setShowAIExplanation] = useState(false);

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
            {/* Header with Explain with AI button */}
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold">Model Overview</h1>
                <ExplainWithAIButton 
                    onClick={() => setShowAIExplanation(true)}
                    size="md"
                />
            </div>

            {/* AI Explanation Panel */}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="overview"
                analysisData={overview}
                title="Model Overview - AI Explanation"
            />

            {/* Rest of the existing content */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
                <MetricCard title="Accuracy" value={metrics.accuracy} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="Precision" value={metrics.precision} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="Recall" value={metrics.recall} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="F1 Score" value={metrics.f1_score} format="percentage" icon={<Target className="w-6 h-6 text-blue-600" />} />
                <MetricCard title="AUC" value={metrics.auc} format="number" icon={<Target className="w-6 h-6 text-blue-600" />} />
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-xl font-semibold mb-4 flex items-center"><Info className="mr-2" /> Model Details</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
                    <div>
                        <ul className="list-disc pl-5 space-y-1">
                            <li><strong>Name:</strong> {overview.name || 'N/A'}</li>
                            <li><strong>Algorithm:</strong> {overview.algorithm || 'N/A'}</li>
                            <li><strong>Version:</strong> {overview.version}</li>
                            <li><strong>Framework:</strong> {overview.framework}</li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-semibold mb-1">Training Information</h3>
                        <ul className="list-disc pl-5 space-y-1">
                            <li><strong>Created:</strong> {overview.metadata?.created}</li>
                            <li><strong>Last Trained:</strong> {overview.metadata?.last_trained}</li>
                            <li><strong>Samples:</strong> {overview.metadata?.samples}</li>
                            <li><strong>Features:</strong> {overview.metadata?.features}</li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-semibold mb-1">Dataset Split</h3>
                        <ul className="list-disc pl-5 space-y-1">
                            <li><strong>Train:</strong> {overview.metadata?.dataset_split?.train}</li>
                            <li><strong>Test:</strong> {overview.metadata?.dataset_split?.test}</li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="font-semibold mb-1">Status</h3>
                        <ul className="list-disc pl-5 space-y-1">
                            <li><strong>Health Score:</strong> {overview.metadata?.health_score_pct?.toFixed?.(0)}%</li>
                            <li><strong>Status:</strong> {overview.status}</li>
                            <li><strong>Duplicates:</strong> {overview.metadata?.duplicates_pct?.toFixed?.(1)}%</li>
                            <li><strong>Missing:</strong> {overview.metadata?.missing_pct?.toFixed?.(1)}%</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ModelOverview;