import React, { useEffect, useState } from 'react';
import { TrendingUp, Search, BarChart3, Settings, AlertCircle, Loader2 } from 'lucide-react';
import { getModelOverview, postPartialDependence, postShapDependence, postIcePlot } from '../../services/api';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

const FeatureCard = ({ name, description, percentage, isSelected, onClick }: {
    name: string;
    description: string;
    percentage: string;
    isSelected: boolean;
    onClick: () => void;
}) => (
    <div
        className={`p-4 rounded-lg cursor-pointer border-2 transition-all ${isSelected
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
            : 'border-gray-200 dark:border-gray-600 hover:border-gray-300'
            }`}
        onClick={onClick}
    >
        <div className="flex items-center justify-between mb-2">
            <span className="font-medium text-sm">{name}</span>
            <span className={`px-2 py-1 text-xs rounded ${isSelected ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600'
                }`}>
                {percentage}
            </span>
        </div>
        <div className="text-xs text-gray-500">{description}</div>
    </div>
);

const PlotTypeSelector = ({ selectedType, onTypeChange }: {
    selectedType: string;
    onTypeChange: (type: string) => void;
}) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Plot Type</h3>
        <div className="grid grid-cols-3 gap-3">
            {[
                { id: 'partial', name: 'Partial Dependence', desc: 'Average effect of feature' },
                { id: 'shap', name: 'SHAP Dependence', desc: 'Feature interaction effects' },
                { id: 'ice', name: 'Individual ICE', desc: 'Individual conditional expectation' }
            ].map(type => (
                <div
                    key={type.id}
                    className={`p-3 rounded-lg cursor-pointer border text-center ${selectedType === type.id
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 hover:border-gray-300'
                        }`}
                    onClick={() => onTypeChange(type.id)}
                >
                    <div className="text-sm font-medium">{type.name}</div>
                    <div className="text-xs text-gray-500 mt-1">{type.desc}</div>
                </div>
            ))}
        </div>
    </div>
);

const PDPPlot = ({ y, feature }: { y: number[]; feature: string }) => {
    const minY = Math.min(...y);
    const maxY = Math.max(...y);
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center"><BarChart3 className="mr-2 text-blue-600" />Partial Dependence: {feature}</h3>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-72">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                    {y.map((yv, i) => {
                        const xNorm = (i / Math.max(1, y.length - 1)) * 100;
                        const yNorm = 100 - ((yv - minY) / (maxY - minY || 1)) * 100;
                        return <circle key={i} cx={xNorm} cy={yNorm} r="1.2" fill="#3b82f6" />;
                    })}
                    {y.slice(1).map((_, i) => {
                        const x1 = (i / Math.max(1, y.length - 1)) * 100;
                        const y1 = 100 - ((y[i] - minY) / (maxY - minY || 1)) * 100;
                        const x2 = ((i + 1) / Math.max(1, y.length - 1)) * 100;
                        const y2 = 100 - ((y[i + 1] - minY) / (maxY - minY || 1)) * 100;
                        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#3b82f6" strokeWidth={1} />;
                    })}
                </svg>
            </div>
        </div>
    );
};

const SHAPDependencePlot = ({ feature_values, shap_values }: { feature_values: (number | string)[]; shap_values: number[] }) => {
    const minX = Math.min(...feature_values.map(v => Number(v)));
    const maxX = Math.max(...feature_values.map(v => Number(v)));
    const minY = Math.min(...shap_values);
    const maxY = Math.max(...shap_values);
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center"><BarChart3 className="mr-2 text-blue-600" />SHAP Dependence</h3>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-72">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                    {feature_values.map((fx, i) => {
                        const x = ((Number(fx) - minX) / (maxX - minX || 1)) * 100;
                        const y = 100 - ((shap_values[i] - minY) / (maxY - minY || 1)) * 100;
                        return <circle key={i} cx={x} cy={y} r="1.1" fill="#3b82f6" opacity={0.7} />;
                    })}
                </svg>
            </div>
        </div>
    );
};

const ICEPlot = ({ curves, feature }: { curves: Array<{ x: (number | string)[]; y: number[] }>; feature: string }) => {
    const allY = curves.flatMap(c => c.y);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center"><BarChart3 className="mr-2 text-blue-600" />ICE Plot: {feature}</h3>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-72">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                    {curves.map((c, ci) => (
                        <g key={ci}>
                            {c.y.slice(1).map((_, i) => {
                                const x1 = (i / Math.max(1, c.y.length - 1)) * 100;
                                const y1 = 100 - ((c.y[i] - minY) / (maxY - minY || 1)) * 100;
                                const x2 = ((i + 1) / Math.max(1, c.y.length - 1)) * 100;
                                const y2 = 100 - ((c.y[i + 1] - minY) / (maxY - minY || 1)) * 100;
                                return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#60a5fa" strokeWidth={0.8} opacity={0.6} />;
                            })}
                        </g>
                    ))}
                </svg>
            </div>
        </div>
    );
};

// removed unused placeholder impact component

const FeatureDependence: React.FC<{ modelType?: string }> = () => {
    const [selectedFeature, setSelectedFeature] = useState('');
    const [plotType, setPlotType] = useState('partial');
    const [searchTerm, setSearchTerm] = useState('');
    const [featureList, setFeatureList] = useState<string[]>([]);
    const [pdp, setPdp] = useState<any>(null);
    const [shapDep, setShapDep] = useState<any>(null);
    const [ice, setIce] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [showAIExplanation, setShowAIExplanation] = useState(false);

    useEffect(() => {
        // load features from overview
        (async () => {
            try {
                const overview = await getModelOverview();
                const names: string[] = overview.feature_names || [];
                setFeatureList(names);
                if (names.length > 0) setSelectedFeature(names[0]);
            } catch (e: any) {
                setError(e.response?.data?.detail || 'Unable to load feature list');
            }
        })();
    }, []);

    useEffect(() => {
        if (!selectedFeature) return;
        (async () => {
            try {
                setLoading(true);
                setError('');
                const [p, s, i] = await Promise.all([
                    postPartialDependence(selectedFeature, 25),
                    postShapDependence(selectedFeature),
                    postIcePlot(selectedFeature, 15, 20)
                ]);
                setPdp(p);
                setShapDep(s);
                setIce(i);
            } catch (e: any) {
                setError(e.response?.data?.detail || 'Failed to load feature dependence');
            } finally {
                setLoading(false);
            }
        })();
    }, [selectedFeature]);

    const filteredFeatures = featureList.filter(f => f.toLowerCase().includes(searchTerm.toLowerCase()));

    return (
        <>
            <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
                <div className="flex items-center justify-between">
                    <h1 className="text-3xl font-bold">Feature Dependence</h1>
                    <div className="flex items-center space-x-4">
                        <p className="text-sm text-gray-500">Explore how individual features affect model predictions across their value ranges</p>
                        <ExplainWithAIButton onClick={() => setShowAIExplanation(true)} size="md" />
                    </div>
                </div>

                <div className="flex space-x-2">
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center space-x-2">
                        <Settings className="w-4 h-4" />
                        <span>Settings</span>
                    </button>
                    <button className="px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600">
                        Export
                    </button>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Feature Selection */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <h2 className="text-lg font-semibold mb-4">Select Feature</h2>

                        <div className="relative mb-4">
                            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search features..."
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-10 pr-4 py-2 border rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
                            />
                        </div>

                        <div className="space-y-3 max-h-64 overflow-y-auto">
                            {filteredFeatures.map((feature) => (
                                <FeatureCard
                                    key={feature}
                                    name={feature}
                                    description={''}
                                    percentage={''}
                                    isSelected={selectedFeature === feature}
                                    onClick={() => setSelectedFeature(feature)}
                                />
                            ))}
                        </div>
                    </div>

                    {/* Plot Display */}
                    <div className="lg:col-span-2">
                        {loading && (
                            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 h-72 flex items-center justify-center">
                                <Loader2 className="w-6 h-6 animate-spin" />
                            </div>
                        )}
                        {error && !loading && (
                            <div className="p-4 text-red-500 bg-red-50 dark:bg-red-900/20 rounded-lg flex items-center">
                                <AlertCircle className="mr-2 flex-shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}
                        {!loading && plotType === 'partial' && pdp && (
                            <PDPPlot y={pdp.y} feature={selectedFeature} />
                        )}
                        {!loading && plotType === 'shap' && shapDep && (
                            <SHAPDependencePlot feature_values={shapDep.feature_values} shap_values={shapDep.shap_values} />
                        )}
                        {!loading && plotType === 'ice' && ice && (
                            <ICEPlot curves={ice.curves} feature={selectedFeature} />
                        )}
                    </div>
                </div>

                {/* Feature Impact Panel based on PDP */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <PlotTypeSelector selectedType={plotType} onTypeChange={setPlotType} />
                    {pdp && (
                        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                            <h3 className="text-lg font-semibold mb-4 flex items-center"><TrendingUp className="mr-2 text-purple-600" />Feature Impact Analysis</h3>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                                    <div className="text-xs text-gray-500">Type</div>
                                    <div className="text-sm font-semibold">{pdp.impact.feature_type}</div>
                                </div>
                                <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                                    <div className="text-xs text-gray-500">Importance</div>
                                    <div className="text-sm font-semibold">{pdp.impact.importance_percentage.toFixed(1)}%</div>
                                </div>
                                <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                                    <div className="text-xs text-gray-500">Effect Range</div>
                                    <div className="text-sm font-semibold">{pdp.impact.effect_range.toFixed(3)}</div>
                                </div>
                                <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                                    <div className="text-xs text-gray-500">Trend</div>
                                    <div className="text-sm font-semibold">{pdp.impact.trend_analysis.direction}</div>
                                </div>
                                <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                                    <div className="text-xs text-gray-500">Confidence</div>
                                    <div className="text-sm font-semibold">{pdp.impact.confidence_score}%</div>
                                </div>
                                <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded">
                                    <div className="text-xs text-gray-500">Summary</div>
                                    <div className="text-sm">{pdp.impact.impact_summary}</div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
                {/* AI Explanation Panel */}
                <AIExplanationPanel
                    isOpen={showAIExplanation}
                    onClose={() => setShowAIExplanation(false)}
                    analysisType="feature_dependence"
                    analysisData={{
                        selectedFeature,
                        plotType,
                        pdp,
                        shapDep,
                        ice
                    }}
                    title="Feature Dependence - AI Explanation"
                />
            </div>
        </>
    );
}; export default FeatureDependence;
