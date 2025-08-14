import React, { useEffect, useMemo, useState } from 'react';
import {
    ResponsiveContainer,
    BarChart as RBarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    RadarChart,
    Radar,
    PolarGrid,
    PolarAngleAxis,
    PolarRadiusAxis,
} from 'recharts';
import { getFeaturesMetadata, postCorrelation, postAdvancedImportance } from '../../services/api';
import { AlertCircle, Loader2, BarChart3, Eye, EyeOff, Info } from 'lucide-react';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

type FeatureMeta = { name: string; type: 'numeric' | 'categorical'; description?: string; min_value?: number; max_value?: number; categories?: string[] };

const heatColor = (v: number) => {
    // v in [-1, 1] -> blue shades
    const x = Math.abs(v);
    const alpha = 0.2 + x * 0.8;
    return `rgba(37, 99, 235, ${alpha})`;
};

const FeatureImportance: React.FC<{ modelType?: string }> = () => {
    // metadata and correlation
    const [featuresMeta, setFeaturesMeta] = useState<FeatureMeta[]>([]);
    const [visible, setVisible] = useState<Record<string, boolean>>({});
    const [corr, setCorr] = useState<{ features: string[]; matrix: number[][] } | null>(null);
    const [corrLoading, setCorrLoading] = useState(false);
    const [error, setError] = useState('');

    // importance controls
    const [viz, setViz] = useState<'bar' | 'radar' | 'table'>('bar');
    const [method, setMethod] = useState<'shap' | 'permutation' | 'gain'>('shap');
    const [sortBy, setSortBy] = useState<'importance' | 'feature_name' | 'impact'>('importance');
    const [topN, setTopN] = useState(10);
    const [impLoading, setImpLoading] = useState(false);
    const [importance, setImportance] = useState<any>(null);

    // AI explanation state
    const [showAIExplanation, setShowAIExplanation] = useState(false);

    // fetch feature metadata
    useEffect(() => {
        (async () => {
            try {
                const meta = await getFeaturesMetadata();
                const list: FeatureMeta[] = meta.features || [];
                setFeaturesMeta(list);
                const v: Record<string, boolean> = {};
                list.forEach(f => (v[f.name] = true));
                setVisible(v);
            } catch (e: any) {
                setError(e.response?.data?.detail || 'Failed to load features metadata');
            }
        })();
    }, []);

    // compute correlation whenever visible set changes
    const activeFeatures = useMemo(() => Object.keys(visible).filter(k => visible[k]), [visible]);

    useEffect(() => {
        if (activeFeatures.length < 2) {
            setCorr(null);
            return;
        }
        (async () => {
            try {
                setCorrLoading(true);
                setError('');
                const res = await postCorrelation(activeFeatures);
                setCorr({ features: res.features, matrix: res.matrix });
            } catch (e: any) {
                setError(e.response?.data?.detail || 'Failed to compute correlation');
            } finally {
                setCorrLoading(false);
            }
        })();
    }, [activeFeatures.join('|')]);

    // fetch advanced importance whenever controls change
    useEffect(() => {
        (async () => {
            try {
                setImpLoading(true);
                setError(''); // Clear previous errors
                const data = await postAdvancedImportance({ method, sort_by: sortBy, top_n: topN, visualization: viz });
                setImportance(data);
                
                // Check if the response contains an error (for SHAP unavailability)
                if (data.error) {
                    setError(data.error);
                    // Auto-fallback to permutation method if SHAP fails
                    if (method === 'shap' && data.error.includes('SHAP values are not available')) {
                        setMethod('gain'); // Try builtin first, will fallback to permutation if needed
                    }
                }
            } catch (e: any) {
                const errorMsg = e.response?.data?.detail || 'Failed to fetch feature importance';
                setError(errorMsg);
                
                // Auto-fallback for SHAP unavailability
                if (method === 'shap' && errorMsg.includes('SHAP')) {
                    setError(errorMsg + ' Automatically switching to builtin method.');
                    setMethod('gain');
                }
            } finally {
                setImpLoading(false);
            }
        })();
    }, [method, sortBy, topN, viz]);

    const toggleFeature = (name: string) => {
        const next = { ...visible };
        const currentlyVisible = Object.values(next).filter(Boolean).length;
        if (next[name] && currentlyVisible <= 2) return; // keep at least two
        next[name] = !next[name];
        setVisible(next);
    };

    const barData = useMemo(() => {
        const arr = importance?.features || [];
        return arr.slice(0, topN).map((f: any) => ({ name: f.name, importance: f.importance_score }));
    }, [importance, topN]);

    const radarData = useMemo(() => {
        return barData.map((d: { name: string; importance: number }) => ({ subject: d.name, A: Math.abs(d.importance) }));
    }, [barData]);

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold flex items-center">
                    <BarChart3 className="mr-3 text-blue-600" />
                    Feature Importance
                </h1>
                <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <Info className="w-4 h-4" />
                        <span>Interactive analysis</span>
                    </div>
                    <ExplainWithAIButton 
                        onClick={() => setShowAIExplanation(true)}
                        size="md"
                    />
                </div>
            </div>

            {/* Controls & Correlation */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Feature Controls */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Feature Controls</h3>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-80 overflow-y-auto">
                        {featuresMeta.map(f => {
                            const on = !!visible[f.name];
                            return (
                                <button key={f.name} className={`flex items-center justify-between px-3 py-2 rounded border text-sm ${on ? 'bg-blue-50 border-blue-300 dark:bg-blue-900/20' : 'bg-gray-50 dark:bg-gray-700 border-gray-300'}`} onClick={() => toggleFeature(f.name)} aria-label={`Toggle feature ${f.name}`}>
                                    <span className="truncate" title={f.name}>{f.name}</span>
                                    {on ? <Eye className="w-4 h-4 text-blue-600" /> : <EyeOff className="w-4 h-4 text-gray-400" />}
                                </button>
                            );
                        })}
                    </div>
                    <div className="text-xs text-gray-500 mt-2">At least two features must remain visible.</div>
                </div>

                {/* Correlation Heatmap */}
                <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <div className="flex items-center justify-between mb-2">
                        <h3 className="text-lg font-semibold">Correlation Matrix</h3>
                        {corrLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                    </div>
                    {!corr && !corrLoading && (
                        <div className="text-sm text-gray-500">Select at least two features to view correlations.</div>
                    )}
                    {corr && (
                        <div className="overflow-auto">
                            <div className="inline-block">
                                {/* labels */}
                                <div className="grid" style={{ gridTemplateColumns: `120px repeat(${corr.features.length}, 1fr)` }}>
                                    <div></div>
                                    {corr.features.map((f, i) => (
                                        <div key={i} className="text-xs text-gray-600 rotate-[-45deg] origin-left h-8 flex items-end" style={{ height: 32 }}>{f}</div>
                                    ))}
                                    {corr.features.map((rowName, r) => (
                                        <React.Fragment key={rowName}>
                                            <div className="text-xs text-gray-600 h-8 flex items-center">{rowName}</div>
                                            {corr.matrix[r].map((v, c) => (
                                                <div key={c} className="h-8 w-16 flex items-center justify-center text-[10px] font-semibold text-white transition-colors" style={{ backgroundColor: heatColor(v) }} title={`${rowName} ↔ ${corr.features[c]}: ${v.toFixed(2)}`}>
                                                    {v.toFixed(2)}
                                                </div>
                                            ))}
                                        </React.Fragment>
                                    ))}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Advanced Importance Controls */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Importance Controls</h3>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                        <label className="text-xs text-gray-500">Visualization</label>
                        <select className="w-full mt-1 px-3 py-2 border rounded bg-white dark:bg-gray-700 dark:border-gray-600" value={viz} onChange={e => setViz(e.target.value as any)}>
                            <option value="bar">Bar</option>
                            <option value="radar">Radar</option>
                            <option value="table">Table</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-gray-500">Method</label>
                        <select className="w-full mt-1 px-3 py-2 border rounded bg-white dark:bg-gray-700 dark:border-gray-600" value={method} onChange={e => setMethod(e.target.value as any)}>
                            <option value="shap">SHAP (may not be available for ONNX)</option>
                            <option value="permutation">Permutation</option>
                            <option value="gain">Gain/Builtin (sklearn models only)</option>
                        </select>
                        <p className="text-xs text-gray-400 mt-1">
                            For ONNX models, try "Permutation" if SHAP is not available
                        </p>
                    </div>
                    <div>
                        <label className="text-xs text-gray-500">Sort By</label>
                        <select className="w-full mt-1 px-3 py-2 border rounded bg-white dark:bg-gray-700 dark:border-gray-600" value={sortBy} onChange={e => setSortBy(e.target.value as any)}>
                            <option value="importance">Importance</option>
                            <option value="feature_name">Feature Name</option>
                            <option value="impact">Impact</option>
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-gray-500">Top N: {topN}</label>
                        <input className="w-full" type="range" min={1} max={20} value={topN} onChange={e => setTopN(parseInt(e.target.value))} />
                    </div>
                </div>
            </div>

            {/* Summary Cards */}
            {importance && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg shadow p-6 text-center">
                        <div className="text-2xl font-bold text-blue-600">{importance.total_features}</div>
                        <div className="text-sm text-gray-600">Total Features</div>
                    </div>
                    <div className="bg-green-50 dark:bg-green-900/20 rounded-lg shadow p-6 text-center">
                        <div className="text-2xl font-bold text-green-600">{importance.positive_impact_count}</div>
                        <div className="text-sm text-gray-600">Positive Impact</div>
                    </div>
                    <div className="bg-red-50 dark:bg-red-900/20 rounded-lg shadow p-6 text-center">
                        <div className="text-2xl font-bold text-red-600">{importance.negative_impact_count}</div>
                        <div className="text-sm text-gray-600">Negative Impact</div>
                    </div>
                </div>
            )}

            {/* Visualization Area */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                {impLoading && <div className="flex items-center justify-center h-48"><Loader2 className="w-6 h-6 animate-spin" /></div>}
                {!impLoading && viz === 'bar' && (
                    <div style={{ width: '100%', height: 360 }}>
                        <ResponsiveContainer>
                            <RBarChart data={barData} layout="vertical" margin={{ left: 20, right: 20 }}>
                                <XAxis type="number" hide />
                                <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 12 }} />
                                <Tooltip formatter={(v: number) => (typeof v === 'number' ? v.toFixed(4) : v)} />
                                <Bar dataKey="importance" fill="#3b82f6" />
                            </RBarChart>
                        </ResponsiveContainer>
                    </div>
                )}
                {!impLoading && viz === 'radar' && (
                    <div style={{ width: '100%', height: 420 }}>
                        <ResponsiveContainer>
                            <RadarChart data={radarData}>
                                <PolarGrid />
                                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 12 }} />
                                <PolarRadiusAxis />
                                <Radar name="Importance" dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                )}
                {!impLoading && viz === 'table' && (
                    <div className="overflow-auto">
                        <table className="min-w-full text-sm">
                            <thead>
                                <tr className="text-left text-gray-500">
                                    <th className="p-2">Rank</th>
                                    <th className="p-2">Feature</th>
                                    <th className="p-2">Importance</th>
                                    <th className="p-2">Impact</th>
                                </tr>
                            </thead>
                            <tbody>
                                {(importance?.features || []).map((f: any) => (
                                    <tr key={f.name} className="border-t border-gray-200 dark:border-gray-700">
                                        <td className="p-2">{f.rank}</td>
                                        <td className="p-2">{f.name}</td>
                                        <td className="p-2 font-mono">{f.importance_score.toFixed(4)}</td>
                                        <td className="p-2">
                                            <span className={`px-2 py-0.5 rounded text-xs ${f.impact_direction === 'positive' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>{f.impact_direction}</span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {error && (
                <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                    <AlertCircle className="mr-2 flex-shrink-0" />
                    <span>{error}</span>
                </div>
            )}

            {/* AI Explanation Panel */}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="feature_importance"
                analysisData={{
                    featuresMeta,
                    importance,
                    correlation: corr,
                    controls: { viz, method, sortBy, topN }
                }}
                title="Feature Importance - AI Explanation"
            />
        </div>
    );
};

export default FeatureImportance;
