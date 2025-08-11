import React, { useState } from 'react';
import { Network, Search, Filter, Settings, BarChart3 } from 'lucide-react';

const InteractionHeatmap = ({ minStrength }: { minStrength: number }) => {
    const features = ['Annual_Inc', 'Credit_Sco', 'Customer_A', 'Account_Ba', 'Loan_Amoun', 'Employment', 'Education'];

    // Mock interaction strength data
    const getInteractionStrength = (i: number, j: number) => {
        if (i === j) return 1.0;
        if ((i === 0 && j === 1) || (i === 1 && j === 0)) return 0.85; // Strong interaction
        if ((i === 0 && j === 2) || (i === 2 && j === 0)) return 0.72;
        if ((i === 1 && j === 2) || (i === 2 && j === 1)) return 0.68;
        const v = Math.random() * 0.6; // Random weak interactions
        return v >= minStrength ? v : v; // keep signature usage
    };

    const getColorIntensity = (strength: number) => {
        if (strength === 1.0) return 'bg-green-500';
        if (strength > 0.8) return 'bg-green-400';
        if (strength > 0.6) return 'bg-yellow-300';
        if (strength > 0.4) return 'bg-orange-300';
        if (strength > 0.2) return 'bg-red-300';
        return 'bg-gray-200';
    };

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Network className="mr-2 text-blue-600" />
                Interaction Heatmap
            </h3>
            <div className="text-sm text-gray-600 mb-4">
                Color intensity represents interaction strength between feature pairs
            </div>

            {/* Heatmap Grid */}
            <div className="grid grid-cols-8 gap-1 mb-4">
                {/* Header row */}
                <div></div>
                {features.map((feature, index) => (
                    <div key={index} className="text-xs text-center font-medium p-2 transform -rotate-45 origin-center">
                        {feature}
                    </div>
                ))}

                {/* Data rows */}
                {features.map((rowFeature, i) => (
                    <React.Fragment key={i}>
                        <div className="text-xs font-medium p-2 flex items-center">
                            {rowFeature}
                        </div>
                        {features.map((colFeature, j) => {
                            const strength = getInteractionStrength(i, j);
                            return (
                                <div
                                    key={j}
                                    className={`aspect-square flex items-center justify-center text-xs font-bold text-white rounded ${getColorIntensity(strength)}`}
                                    title={`${rowFeature} × ${colFeature}: ${strength.toFixed(2)}`}
                                >
                                    {strength === 1.0 ? '100' : (strength * 100).toFixed(0)}
                                </div>
                            );
                        })}
                    </React.Fragment>
                ))}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center space-x-4 text-xs">
                <span>Min Strength:</span>
                <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-gray-200 rounded"></div>
                    <span>0.10</span>
                </div>
                <div className="flex items-center space-x-2">
                    <div className="w-4 h-4 bg-green-500 rounded"></div>
                    <span>Max</span>
                </div>
            </div>
        </div>
    );
};

const AnalysisControls = ({ minStrength, onMinStrengthChange }: {
    minStrength: number;
    onMinStrengthChange: (value: number) => void;
}) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Filter className="mr-2" />
            Analysis Controls
        </h3>

        <div className="space-y-4">
            <div className="grid grid-cols-3 gap-2">
                <button className="px-3 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700">
                    Interaction Heatmap
                </button>
                <button className="px-3 py-2 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300">
                    Network Graph
                </button>
                <button className="px-3 py-2 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300">
                    Pairwise Analysis
                </button>
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
                    placeholder="Search feature pairs..."
                    className="w-full pl-10 pr-4 py-2 border rounded-md bg-white dark:bg-gray-700 dark:border-gray-600"
                />
            </div>
        </div>
    </div>
);

const TopInteractions = () => {
    const interactions = [
        { feature1: 'Annual_Income', feature2: 'Credit_Score', strength: 0.85, description: 'Strong positive correlation' },
        { feature1: 'Customer_Age', feature2: 'Account_Balance', strength: 0.72, description: 'Moderate interaction effect' },
        { feature1: 'Employment_Type', feature2: 'Education_Level', strength: 0.68, description: 'Career-related synergy' }
    ];

    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
                <BarChart3 className="mr-2 text-purple-600" />
                Top Interactions
            </h3>

            <div className="space-y-4">
                {interactions.map((interaction, index) => (
                    <div key={index} className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <div className="font-medium text-sm">
                                {interaction.feature1} × {interaction.feature2}
                            </div>
                            <div className="text-lg font-bold text-blue-600">
                                {(interaction.strength * 100).toFixed(0)}%
                            </div>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2 mb-2">
                            <div
                                className="h-2 bg-blue-500 rounded-full"
                                style={{ width: `${interaction.strength * 100}%` }}
                            ></div>
                        </div>
                        <div className="text-xs text-gray-500">{interaction.description}</div>
                    </div>
                ))}
            </div>
        </div>
    );
};

const FeatureInteractions: React.FC<{ modelType?: string }> = () => {
    const [minStrength, setMinStrength] = useState(0.1);

    return (
        <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Feature Interactions</h1>
                <p className="text-sm text-gray-500">Discover how features interact with each other to influence predictions</p>
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
                    minStrength={minStrength}
                    onMinStrengthChange={setMinStrength}
                />
                <div className="lg:col-span-3">
                    <InteractionHeatmap minStrength={minStrength} />
                </div>
            </div>

            <TopInteractions />
        </div>
    );
}; export default FeatureInteractions;
