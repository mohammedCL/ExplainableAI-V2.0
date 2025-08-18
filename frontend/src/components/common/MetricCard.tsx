import React from 'react';

const MetricCard: React.FC<{ title: string; value: number | undefined; format: 'percentage' | 'number'; }> = ({ title, value, format }) => {
    const formatValue = (val: number | undefined, fmt: 'percentage' | 'number') => {
        if (val === undefined || val === null || isNaN(val)) {
            return 'N/A';
        }
        return fmt === 'percentage' ? `${(val * 100).toFixed(1)}%` : val.toFixed(3);
    };

    return (
        <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
            <h3 className="text-lg font-bold">{title}</h3>
            <p className="text-2xl">
                {formatValue(value, format)}
            </p>
        </div>
    );
};

export default MetricCard;
