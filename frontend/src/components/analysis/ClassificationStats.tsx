import React, { useEffect, useMemo, useState } from 'react';
import { getClassificationStats, postRocAnalysis, postThresholdAnalysis } from '../../services/api';
import { AlertCircle, Loader2, PieChart, Target, TrendingUp, CheckCircle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

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
    // Place ALL hooks at top-level and in consistent order across renders
    const [stats, setStats] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [roc, setRoc] = useState<any>(null);
    const [thr, setThr] = useState<any>(null);

    // Initial data fetches
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

    useEffect(() => {
        (async () => {
            try {
                const [r, t] = await Promise.all([postRocAnalysis(), postThresholdAnalysis(50)]);
                setRoc(r);
                setThr(t);
            } catch { /* handled visually */ }
        })();
    }, []);

    // Derived data guarded for nulls so hooks are called every render
    const metrics = (stats?.metrics ?? { accuracy: 0, precision: 0, recall: 0, f1_score: 0, auc: 0 });
    const confusion_matrix = (stats?.confusion_matrix ?? { true_negative: 0, false_positive: 0, false_negative: 0, true_positive: 0 });

    const rocData = useMemo(() => {
        if (!roc) return [] as any[];
        return roc.roc_curve.fpr.map((f: number, i: number) => ({ fpr: f, tpr: roc.roc_curve.tpr[i] }));
    }, [roc]);
    const diagData = useMemo(() => ([{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]), []);

    // Only after hooks we conditionally render
    if (loading) {
        return <div className="p-6 flex justify-center items-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>;
    }
    if (error) {
        return <div className="p-6 text-red-500"><AlertCircle className="inline-block mr-2" />{error}</div>;
    }
    if (!stats) {
        return <div className="p-6">No classification stats available.</div>;
    }

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
            <div className="p-0 sm:p-2 md:p-4 lg:p-6 max-w-none">
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

                    {/* Enhanced ROC Curve */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-xl font-semibold mb-4 flex items-center"><TrendingUp className="mr-2" /> ROC Analysis</h2>
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="lg:col-span-2" style={{ width: '100%', height: 320 }}>
                                {roc ? (
                                    <ResponsiveContainer>
                                        <LineChart data={rocData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} />
                                            <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => Number(v).toFixed(1)} />
                                            <Tooltip formatter={(v: number) => Number(v).toFixed(3)} />
                                            <Line data={diagData} dataKey="tpr" stroke="#9CA3AF" strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                                            <Line type="monotone" dataKey="tpr" stroke="#3b82f6" dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-sm text-gray-500">
                                        <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading ROC analysis
                                    </div>
                                )}
                            </div>
                            <div className="space-y-3">
                                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded">
                                    <div className="text-xs text-gray-500">AUC Score</div>
                                    <div className="text-2xl font-bold text-blue-600">{(roc?.metrics?.auc_score ?? metrics.auc).toFixed(3)}</div>
                                </div>
                                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded">
                                    <div className="text-xs text-gray-500">Optimal Threshold</div>
                                    <div className="text-2xl font-bold text-purple-600">{(roc?.metrics?.optimal_threshold ?? 0.5).toFixed(2)}</div>
                                </div>
                                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                                    <div className="text-xs text-gray-500">Sensitivity at Optimal</div>
                                    <div className="text-2xl font-bold text-green-600">{(roc?.metrics?.sensitivity ?? metrics.recall).toFixed(2)}</div>
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

                        {/* Classification Threshold Analysis (diagonal heatmap style) */}
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                            <h3 className="text-lg font-semibold mb-4">Threshold Analysis</h3>
                            <div style={{ width: '100%', height: 280 }}>
                                {thr ? (
                                    <ResponsiveContainer>
                                        <LineChart data={(thr?.threshold_metrics || []).map((m: any) => ({ x: m.threshold, precision: m.precision, recall: m.recall, f1: m.f1_score, acc: m.accuracy }))}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="x" tickFormatter={(v) => Number(v).toFixed(2)} />
                                            <YAxis domain={[0, 1]} />
                                            <Tooltip formatter={(v: number) => Number(v).toFixed(3)} labelFormatter={(v) => `Threshold: ${Number(v).toFixed(2)}`} />
                                            <Line type="monotone" dataKey="precision" stroke="#ef4444" dot={false} />
                                            <Line type="monotone" dataKey="recall" stroke="#22c55e" dot={false} />
                                            <Line type="monotone" dataKey="f1" stroke="#a855f7" dot={false} />
                                            <Line type="monotone" dataKey="acc" stroke="#3b82f6" dot={false} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-sm text-gray-500">
                                        <Loader2 className="w-5 h-5 animate-spin mr-2" /> Loading threshold analysis
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ClassificationStats;
