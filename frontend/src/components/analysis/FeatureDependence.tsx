import React, { useEffect, useState } from 'react';
import { TrendingUp, Search, BarChart3, Settings, AlertCircle, Loader2 } from 'lucide-react';
import { getFeatureDependence, getModelOverview } from '../../services/api';

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

const DependencePlot = ({ feature, featureValues, shapValues }: { feature: string; featureValues: number[]; shapValues: number[] }) => {
    const min = Math.min(...shapValues);
    const max = Math.max(...shapValues);
    const range = max - min;
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
                <BarChart3 className="mr-2 text-blue-600" />
                Partial Dependence: {feature}
            </h3>

            {/* simple SVG scatter plot */}
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-72">
                <svg viewBox="0 0 100 100" className="w-full h-full">
                    {featureValues.map((fv, i) => {
                        const x = ((fv - Math.min(...featureValues)) / (Math.max(...featureValues) - Math.min(...featureValues) || 1)) * 100;
                        const y = 100 - ((shapValues[i] - min) / (range || 1)) * 100;
                        return <circle key={i} cx={x} cy={y} r="1.2" fill="#3b82f6" opacity="0.8" />;
                    })}
                </svg>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-4">
                <div className="text-center">
                    <div className="text-lg font-bold text-blue-600">{min.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Min Effect</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-green-600">{max.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Max Effect</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-purple-600">{range.toFixed(3)}</div>
                    <div className="text-xs text-gray-500">Range</div>
                </div>
            </div>
        </div>
    );
};

const FeatureImpactAnalysis = () => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="mr-2 text-purple-600" />
            Feature Impact Analysis
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Analysis of how the selected feature impacts model predictions
        </div>

        <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <div className="font-medium text-sm mb-2">Key Insights</div>
                <ul className="text-xs text-gray-600 space-y-1">
                    <li>• Feature shows strong positive correlation with predictions</li>
                    <li>• Impact increases significantly at higher values</li>
                    <li>• Most influential range: 60k - 120k</li>
                </ul>
            </div>

            <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                    <div className="text-sm font-medium">Strong Impact</div>
                    <div className="text-xs text-gray-500 mt-1">Above 75th percentile</div>
                </div>
                <div className="text-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <div className="text-sm font-medium">Moderate Impact</div>
                    <div className="text-xs text-gray-500 mt-1">25th - 75th percentile</div>
                </div>
            </div>
        </div>
    </div>
);

const FeatureDependence: React.FC<{ modelType?: string }> = () => {
    const [selectedFeature, setSelectedFeature] = useState('');
    const [plotType, setPlotType] = useState('partial');
    const [searchTerm, setSearchTerm] = useState('');
    const [featureList, setFeatureList] = useState<string[]>([]);
    const [plotData, setPlotData] = useState<{ feature_values: number[], shap_values: number[] } | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

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
                const data = await getFeatureDependence(selectedFeature);
                setPlotData(data);
            } catch (e: any) {
                setError(e.response?.data?.detail || 'Failed to load feature dependence');
            } finally {
                setLoading(false);
            }
        })();
    }, [selectedFeature]);

    const filteredFeatures = featureList.filter(f => f.toLowerCase().includes(searchTerm.toLowerCase()));

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 w-full">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Feature Dependence</h1>
                <p className="text-sm text-gray-500">Explore how individual features affect model predictions across their value ranges</p>
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
                    {plotData && !loading && (
                        <DependencePlot feature={selectedFeature} featureValues={plotData.feature_values} shapValues={plotData.shap_values} />
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PlotTypeSelector selectedType={plotType} onTypeChange={setPlotType} />
                <FeatureImpactAnalysis />
            </div>
        </div>
    );
}; export default FeatureDependence;
