import React from 'react';

const MetricCard: React.FC<{ title: string; value: number; format: 'percentage' | 'number'; }> = ({ title, value, format }) => {
    return (
        <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-lg font-bold">{title}</h3>
            <p className="text-2xl">
                {format === 'percentage' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)}
            </p>
        </div>
    );
};

export default MetricCard;
