import React, { useEffect, useState } from 'react';
import { Network, Search, Filter, Settings, BarChart3, Loader2 } from 'lucide-react';
import { getModelOverview, postInteractionNetwork, postPairwiseAnalysis } from '../../services/api';
import Plot from 'react-plotly.js';
// @ts-ignore - types are provided by package at runtime
import ForceGraph2D from 'react-force-graph-2d';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

// Removed legacy heatmap demo

const AnalysisControls = ({ active, onActiveChange, minStrength, onMinStrengthChange, searchText, onSearchTextChange, onSearchSubmit }: {
    active: 'heatmap' | 'network' | 'pairwise';
    onActiveChange: (v: 'heatmap' | 'network' | 'pairwise') => void;
    minStrength: number;
    onMinStrengthChange: (value: number) => void;
    searchText: string;
    onSearchTextChange: (value: string) => void;
    onSearchSubmit: () => void;
}) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Filter className="mr-2" />
            Analysis Controls
        </h3>

        <div className="space-y-4">
            <div className="grid grid-cols-3 gap-2">
                <button className={`px-3 py-2 rounded text-sm ${active === 'heatmap' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`} onClick={() => onActiveChange('heatmap')}>Interaction Heatmap</button>
                <button className={`px-3 py-2 rounded text-sm ${active === 'network' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`} onClick={() => onActiveChange('network')}>Network Graph</button>
                <button className={`px-3 py-2 rounded text-sm ${active === 'pairwise' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`} onClick={() => onActiveChange('pairwise')}>Pairwise Analysis</button>
            </div>

            <div>
                <div className="flex justify-between items-center mb-2">
                    <label className="text-sm font-medium">Min Strength:</label>
                    <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 bg-red-500 rounded"></div>
                        <span className="text-sm">{minStrength.toFixed(2)}</span>
                    </div>
                </div>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={minStrength}
                    onChange={(e) => onMinStrengthChange(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                <input
                    type="text"
                    placeholder="Search pair e.g. age-credit_score or age,credit_score"
                    value={searchText}
                    onChange={(e) => onSearchTextChange(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') onSearchSubmit(); }}
                    className="w-full pl-10 pr-4 py-2 border rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
                />
            </div>
        </div>
    </div>
);

const TopInteractions = ({ items, onSelect }: { items: Array<{ feature_pair: string[]; interaction_score: number; classification: string }>; onSelect: (a: string, b: string) => void }) => {
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
                <BarChart3 className="mr-2 text-purple-600" />
                Top Interactions
            </h3>

            <div className="space-y-4">
                {items.map((interaction, index) => (
                    <div key={index} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg cursor-pointer" onClick={() => onSelect(interaction.feature_pair[0], interaction.feature_pair[1])}>
                        <div className="flex justify-between items-center mb-2">
                            <div className="font-medium text-sm">
                                {interaction.feature_pair[0]} × {interaction.feature_pair[1]}
                            </div>
                            <div className="text-lg font-bold text-blue-600">
                                {interaction.interaction_score.toFixed(3)}
                            </div>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mb-2">
                            <div
                                className="h-2 bg-blue-500 rounded-full"
                                style={{ width: `${Math.min(100, interaction.interaction_score * 100)}%` }}
                            ></div>
                        </div>
                        <div className="text-xs text-gray-500">{interaction.classification}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const FeatureInteractions: React.FC<{ modelType?: string }> = () => {
    const [minStrength, setMinStrength] = useState(0.1);
    const [network, setNetwork] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    // keep state for future error UI but unused now
    // error state removed to avoid unused warning
    const [pair, setPair] = useState<{ f1: string; f2: string } | null>(null);
    const [pairData, setPairData] = useState<any>(null);
    const [active, setActive] = useState<'heatmap' | 'network' | 'pairwise'>('heatmap');
    const [searchText, setSearchText] = useState('');
    const [showAIExplanation, setShowAIExplanation] = useState(false);

    useEffect(() => {
        (async () => {
            try {
                await getModelOverview();
                setLoading(true);
                const net = await postInteractionNetwork(30, 200);
                setNetwork(net);
                if (net?.top_interactions?.length) {
                    const [a, b] = net.top_interactions[0].feature_pair;
                    setPair({ f1: a, f2: b });
                } else if (Array.isArray(net?.matrix_features) && net.matrix_features.length >= 2) {
                    setPair({ f1: net.matrix_features[0], f2: net.matrix_features[1] });
                } else if (Array.isArray(net?.feature_names) && net.feature_names.length >= 2) {
                    setPair({ f1: net.feature_names[0], f2: net.feature_names[1] });
                }
            } catch (e: any) {
                console.warn(e);
            } finally {
                setLoading(false);
            }
        })();
    }, []);

    useEffect(() => {
        (async () => {
            if (!pair) return;
            try {
                setLoading(true);
                const data = await postPairwiseAnalysis(pair.f1, pair.f2);
                setPairData(data);
            } catch (e: any) {
                console.warn(e);
            } finally {
                setLoading(false);
            }
        })();
    }, [pair?.f1, pair?.f2]);

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Feature Interactions</h1>
                <div className="flex items-center space-x-4">
                    <p className="text-sm text-gray-500">Discover how features interact with each other to influence predictions</p>
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

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <AnalysisControls
                    active={active}
                    onActiveChange={(v) => {
                        setActive(v);
                        if (v === 'pairwise') {
                            if (network?.top_interactions?.length) {
                                const [a, b] = network.top_interactions[0].feature_pair;
                                setPair({ f1: a, f2: b });
                            } else if (Array.isArray(network?.matrix_features) && network.matrix_features.length >= 2) {
                                setPair({ f1: network.matrix_features[0], f2: network.matrix_features[1] });
                            } else if (Array.isArray(network?.feature_names) && network.feature_names.length >= 2) {
                                setPair({ f1: network.feature_names[0], f2: network.feature_names[1] });
                            }
                        }
                    }}
                    minStrength={minStrength}
                    onMinStrengthChange={setMinStrength}
                    searchText={searchText}
                    onSearchTextChange={setSearchText}
                    onSearchSubmit={() => {
                        const tokens = searchText.split(/[^A-Za-z0-9_]+/).filter(Boolean);
                        if (tokens.length >= 2) {
                            const a = tokens[0];
                            const b = tokens[1];
                            setPair({ f1: a, f2: b });
                            setActive('pairwise');
                        }
                    }}
                />
                <div className="lg:col-span-3">
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold flex items-center"><Network className="mr-2 text-blue-600" />Interaction {active === 'heatmap' ? 'Heatmap' : active === 'network' ? 'Network' : 'Pairwise'}</h3>
                            {network?.summary && (
                                <div className="text-xs text-gray-500">
                                    Edges: {network.summary.total_edges || network.summary.total_interactions || 0} · 
                                    Mean: {(network.summary.mean_strength || network.summary.avg_interaction_strength || 0).toFixed(3)} · 
                                    Median: {(network.summary.median_strength || 0).toFixed(3)}
                                </div>
                            )}
                        </div>
                        {loading && <div className="h-64 flex items-center justify-center"><Loader2 className="w-5 h-5 animate-spin" /></div>}
                        {!loading && network && active === 'heatmap' && (
                            <div className="overflow-auto">
                                <Plot
                                    data={[{
                                        z: (network.matrix || network.interaction_matrix) as number[][],
                                        x: network.matrix_features || network.feature_names,
                                        y: network.matrix_features || network.feature_names,
                                        type: 'heatmap',
                                        colorscale: [[0, '#eef2ff'], [1, '#16a34a']],
                                        zmin: 0,
                                        zmax: 1,
                                        showscale: true,
                                        hovertemplate: '%{y} × %{x}: %{z:.2f}<extra></extra>'
                                    }]}
                                    layout={{
                                        autosize: true,
                                        margin: { l: 80, r: 20, t: 10, b: 80 },
                                        xaxis: { tickangle: -45 },
                                        yaxis: { autorange: 'reversed' },
                                        height: 420
                                    }}
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>
                        )}
                        {!loading && network && active === 'network' && (
                            <div className="h-[420px]">
                                <ForceGraph2D
                                    graphData={{
                                        nodes: (network.nodes || []).map((n: any) => ({ id: n.id, name: n.name, val: 1 + (n.importance || 0) })),
                                        links: (network.edges || []).filter((e: any) => e.strength >= minStrength).map((e: any) => ({ source: e.source, target: e.target, value: e.strength }))
                                    }}
                                    nodeAutoColorBy="id"
                                    nodeRelSize={6}
                                    linkDirectionalParticles={0}
                                    linkColor={() => 'rgba(59,130,246,0.6)'}
                                    linkWidth={(l: any) => Math.max(0.5, (l.value || 0) * 3)}
                                    d3VelocityDecay={0.3}
                                />
                            </div>
                        )}
                        {!loading && active === 'pairwise' && pairData && (
                            <div className="h-[420px]">
                                <Plot
                                    data={[{
                                        x: pairData.x,
                                        y: pairData.y,
                                        mode: 'markers',
                                        type: 'scatter',
                                        marker: { size: 6, color: pairData.prediction, colorscale: 'Blues', colorbar: { title: { text: 'Pred' } } },
                                        hovertemplate: `${pair?.f1}: %{x}<br>${pair?.f2}: %{y}<br>p: %{marker.color:.3f}<extra></extra>`
                                    }]}
                                    layout={{ autosize: true, margin: { l: 40, r: 20, t: 10, b: 40 }, xaxis: { title: { text: pair?.f1 || '' } }, yaxis: { title: { text: pair?.f2 || '' } } }}
                                    style={{ width: '100%', height: '100%' }}
                                    config={{ displayModeBar: false }}
                                />
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {network && (
                <TopInteractions items={network.top_interactions || []} onSelect={(a, b) => { setPair({ f1: a, f2: b }); setActive('pairwise'); }} />
            )}

            {/* Pairwise Analysis */}
            {pairData && Array.isArray(pairData.x) && Array.isArray(pairData.y) && pairData.x.every((v: any) => typeof v === 'number') && pairData.y.every((v: any) => typeof v === 'number') && (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Pairwise Analysis: {pair?.f1} × {pair?.f2}</h3>
                    <div className="text-xs text-gray-500 mb-2">Points colored implicitly by prediction</div>
                    <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4 h-72">
                        <svg viewBox="0 0 100 100" className="w-full h-full">
                            {pairData.x.map((_: any, i: number) => {
                                const xn = ((Number(pairData.x[i]) - Math.min(...pairData.x.map((v: any) => Number(v)))) / (Math.max(...pairData.x.map((v: any) => Number(v))) - Math.min(...pairData.x.map((v: any) => Number(v))) || 1)) * 100;
                                const yn = 100 - ((Number(pairData.y[i]) - Math.min(...pairData.y.map((v: any) => Number(v)))) / (Math.max(...pairData.y.map((v: any) => Number(v))) - Math.min(...pairData.y.map((v: any) => Number(v))) || 1)) * 100;
                                const p = pairData.prediction[i];
                                const blue = Math.round(120 + 135 * p);
                                return <circle key={i} cx={xn} cy={yn} r="1.2" fill={`rgb(30,144,${blue})`} opacity={0.75} />;
                            })}
                        </svg>
                    </div>
                </div>
            )}

            {network && (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                    <h3 className="text-lg font-semibold mb-4">Interaction Summary</h3>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                            <div className="text-xs text-gray-500">Total Edges</div>
                            <div className="text-lg font-bold">{network.summary?.total_edges ?? network.summary?.total_interactions ?? 0}</div>
                        </div>
                        <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded">
                            <div className="text-xs text-gray-500">Mean Strength</div>
                            <div className="text-lg font-bold">{(network.summary?.mean_strength ?? network.summary?.avg_interaction_strength ?? 0).toFixed(3)}</div>
                        </div>
                        <div className="p-3 bg-purple-50 dark:bg-purple-900/20 rounded">
                            <div className="text-xs text-gray-500">Median Strength</div>
                            <div className="text-lg font-bold">{(network.summary?.median_strength ?? 0).toFixed(3)}</div>
                        </div>
                        <div className="p-3 bg-orange-50 dark:bg-orange-900/20 rounded">
                            <div className="text-xs text-gray-500">Independence Ratio</div>
                            <div className="text-lg font-bold">{((network.summary?.independence_ratio ?? 0) * 100).toFixed(0)}%</div>
                        </div>
                    </div>
                    {network.top_features && (
                        <div className="mt-4">
                            <div className="text-sm font-semibold mb-2">Top Features</div>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 text-xs">
                                {network.top_features.map((f: any, i: number) => (
                                    <div key={i} className="flex justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded">
                                        <span>{f.name}</span>
                                        <span className="font-mono">{f.importance.toFixed(3)}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="feature_interactions"
                analysisData={{
                    minStrength,
                    network,
                    active,
                    pair,
                    pairData,
                    searchText
                }}
                title="Feature Interactions - AI Explanation"
            />
        </div>
    );
}; export default FeatureInteractions;
