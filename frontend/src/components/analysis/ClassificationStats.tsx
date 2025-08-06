import React, { useState, useEffect } from 'react';
import { getClassificationStats } from '../../services/api';
import { AlertCircle, Loader2, PieChart, Target, TrendingUp, CheckCircle } from 'lucide-react';

const MetricCard = ({ title, value, format, icon, color = "blue" }: {
    title: string;
    value: number;
    format: 'percentage' | 'number';
    icon: React.ReactNode;
    color?: string;
}) => (
    <div className="p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border-l-4 border-blue-500">
        <div className="flex items-center justify-between">
            <div>
                <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{title}</p>
                <p className="text-3xl font-bold text-gray-900 dark:text-white">
                    {format === 'percentage' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
                </p>
            </div>
            <div className={`p-3 bg-${color}-100 dark:bg-${color}-900/50 rounded-full`}>
                {icon}
            </div>
        </div>
    </div>
);

const ConfusionMatrixCell = ({ label, value, isCorrect }: { label: string; value: number; isCorrect: boolean }) => (
    <div className={`p-4 rounded-lg text-center ${isCorrect ? 'bg-green-100 dark:bg-green-900/30' : 'bg-red-100 dark:bg-red-900/30'}`}>
        <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">{label}</div>
        <div className="text-2xl font-bold">{value}</div>
    </div>
);

const ClassificationStats: React.FC = () => {
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchStats = async () => {
            try {
                setLoading(true);
                const data = await getClassificationStats();
                setStats(data);
            } catch (err: any) {
                setError(err.response?.data?.detail || 'Failed to fetch classification stats.');
            } finally {
                setLoading(false);
            }
        };
        fetchStats();
    }, []);

    if (loading) {
        return <div className="p-6 flex justify-center items-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>;
    }
    if (error) {
        return <div className="p-6 text-red-500"><AlertCircle className="inline-block mr-2" />{error}</div>;
    }
    if (!stats) {
        return <div className="p-6">No classification stats available.</div>;
    }

    const { metrics, confusion_matrix, roc_curve } = stats;

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="p-6 max-w-full">
                <div className="space-y-6">
                    <div className="flex items-center justify-between">
                        <h1 className="text-3xl font-bold flex items-center">
                            <PieChart className="mr-3 text-blue-600" />
                            Classification Performance
                        </h1>
                    </div>

                    {/* Performance Metrics */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <MetricCard
                            title="Accuracy"
                            value={metrics.accuracy}
                            format="percentage"
                            icon={<Target className="w-6 h-6 text-blue-600" />}
                        />
                        <MetricCard
                            title="Precision"
                            value={metrics.precision}
                            format="percentage"
                            icon={<CheckCircle className="w-6 h-6 text-green-600" />}
                            color="green"
                        />
                        <MetricCard
                            title="Recall"
                            value={metrics.recall}
                            format="percentage"
                            icon={<TrendingUp className="w-6 h-6 text-purple-600" />}
                            color="purple"
                        />
                        <MetricCard
                            title="F1 Score"
                            value={metrics.f1_score}
                            format="percentage"
                            icon={<Target className="w-6 h-6 text-orange-600" />}
                            color="orange"
                        />
                    </div>

                    {/* Confusion Matrix */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-xl font-semibold mb-6 flex items-center">
                            <PieChart className="mr-2" />
                            Confusion Matrix
                        </h2>
                        <div className="grid grid-cols-2 gap-4 max-w-md mx-auto">
                            <div className="text-center">
                                <div className="text-sm font-medium text-gray-500 mb-4">Predicted</div>
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="text-xs text-gray-400">Negative</div>
                                    <div className="text-xs text-gray-400">Positive</div>
                                </div>
                            </div>
                            <div></div>
                            <div className="flex items-center">
                                <div className="text-sm font-medium text-gray-500 -rotate-90 w-8">Actual</div>
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                                <ConfusionMatrixCell
                                    label="TN"
                                    value={confusion_matrix.true_negative}
                                    isCorrect={true}
                                />
                                <ConfusionMatrixCell
                                    label="FP"
                                    value={confusion_matrix.false_positive}
                                    isCorrect={false}
                                />
                                <ConfusionMatrixCell
                                    label="FN"
                                    value={confusion_matrix.false_negative}
                                    isCorrect={false}
                                />
                                <ConfusionMatrixCell
                                    label="TP"
                                    value={confusion_matrix.true_positive}
                                    isCorrect={true}
                                />
                            </div>
                        </div>
                        <div className="mt-4 text-center text-sm text-gray-500">
                            <div className="flex justify-center space-x-6">
                                <div className="flex items-center">
                                    <div className="w-3 h-3 bg-green-200 rounded mr-2"></div>
                                    Correct Predictions
                                </div>
                                <div className="flex items-center">
                                    <div className="w-3 h-3 bg-red-200 rounded mr-2"></div>
                                    Incorrect Predictions
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ROC Curve Data */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-xl font-semibold mb-4 flex items-center">
                            <TrendingUp className="mr-2" />
                            ROC Curve Information
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <h3 className="font-medium mb-2">AUC Score</h3>
                                <div className="text-3xl font-bold text-blue-600">
                                    {metrics.auc.toFixed(3)}
                                </div>
                                <p className="text-sm text-gray-500 mt-1">
                                    Area Under the Curve
                                </p>
                            </div>
                            <div>
                                <h3 className="font-medium mb-2">Interpretation</h3>
                                <div className="text-sm text-gray-600 dark:text-gray-400">
                                    {metrics.auc > 0.9 && "Excellent performance"}
                                    {metrics.auc > 0.8 && metrics.auc <= 0.9 && "Good performance"}
                                    {metrics.auc > 0.7 && metrics.auc <= 0.8 && "Fair performance"}
                                    {metrics.auc > 0.6 && metrics.auc <= 0.7 && "Poor performance"}
                                    {metrics.auc <= 0.6 && "Very poor performance"}
                                </div>
                                <div className="mt-2 text-xs text-gray-500">
                                    ROC data points: {roc_curve.fpr.length}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Additional Analysis */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        {/* Model Performance Summary */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                            <h3 className="text-lg font-semibold mb-4">Performance Summary</h3>
                            <div className="space-y-4">
                                <div className="flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                                    <span className="text-sm font-medium">Total Samples</span>
                                    <span className="font-bold">
                                        {confusion_matrix.true_positive + confusion_matrix.true_negative +
                                            confusion_matrix.false_positive + confusion_matrix.false_negative}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                                    <span className="text-sm font-medium">Correct Predictions</span>
                                    <span className="font-bold text-green-600">
                                        {confusion_matrix.true_positive + confusion_matrix.true_negative}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                                    <span className="text-sm font-medium">Incorrect Predictions</span>
                                    <span className="font-bold text-red-600">
                                        {confusion_matrix.false_positive + confusion_matrix.false_negative}
                                    </span>
                                </div>
                                <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                                    <span className="text-sm font-medium">Error Rate</span>
                                    <span className="font-bold">
                                        {(((confusion_matrix.false_positive + confusion_matrix.false_negative) /
                                            (confusion_matrix.true_positive + confusion_matrix.true_negative +
                                                confusion_matrix.false_positive + confusion_matrix.false_negative)) * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Classification Threshold Analysis */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                            <h3 className="text-lg font-semibold mb-4">Threshold Analysis</h3>
                            <div className="space-y-4">
                                <div className="text-center mb-4">
                                    <div className="text-2xl font-bold text-purple-600">0.50</div>
                                    <div className="text-sm text-gray-500">Current Threshold</div>
                                </div>
                                <div className="space-y-3">
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm">Sensitivity (TPR)</span>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                                <div
                                                    className="h-2 bg-purple-500 rounded-full"
                                                    style={{ width: `${metrics.recall * 100}%` }}
                                                ></div>
                                            </div>
                                            <span className="text-sm font-medium w-12">{(metrics.recall * 100).toFixed(0)}%</span>
                                        </div>
                                    </div>
                                    <div className="flex justify-between items-center">
                                        <span className="text-sm">Specificity (TNR)</span>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-20 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                                                <div
                                                    className="h-2 bg-green-500 rounded-full"
                                                    style={{
                                                        width: `${((confusion_matrix.true_negative /
                                                            (confusion_matrix.true_negative + confusion_matrix.false_positive)) * 100)}%`
                                                    }}
                                                ></div>
                                            </div>
                                            <span className="text-sm font-medium w-12">
                                                {((confusion_matrix.true_negative /
                                                    (confusion_matrix.true_negative + confusion_matrix.false_positive)) * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ClassificationStats;
