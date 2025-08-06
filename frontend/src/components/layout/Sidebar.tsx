import React from 'react';
import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard, BarChart3, PieChart, UserCheck, TestTube2,
    GitBranch, Share2, Binary
} from 'lucide-react';

const navItems = [
    { name: 'Model Overview', path: '/overview', icon: LayoutDashboard, status: 'complete' },
    { name: 'Feature Importance', path: '/feature-importance', icon: BarChart3, status: 'complete' },
    { name: 'Classification Stats', path: '/classification-stats', icon: PieChart, status: 'complete' },
    { name: 'Individual Predictions', path: '/individual-predictions', icon: UserCheck, status: 'in-progress' },
    { name: 'What-If Analysis', path: '/what-if', icon: TestTube2, status: 'pending' },
    { name: 'Feature Dependence', path: '/feature-dependence', icon: GitBranch, status: 'pending' },
    { name: 'Feature Interactions', path: '/feature-interactions', icon: Share2, status: 'pending' },
    { name: 'Decision Trees', path: '/decision-trees', icon: Binary, status: 'pending' },
];

const StatusIcon = ({ status }: { status: string }) => {
    switch (status) {
        case 'complete':
            return <div className="w-2.5 h-2.5 bg-green-500 rounded-full" />;
        case 'in-progress':
            return <div className="w-2.5 h-2.5 bg-yellow-500 rounded-full animate-pulse" />;
        default: // pending
            return <div className="w-2.5 h-2.5 bg-gray-300 dark:bg-gray-600 rounded-full" />;
    }
};

const Sidebar: React.FC = () => {
    return (
        <div className="flex flex-col w-64 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                <h1 className="text-xl font-bold text-gray-800 dark:text-white">Model Analysis</h1>
                <p className="text-sm text-gray-500 dark:text-gray-400">Classification v1.2</p>
            </div>
            <nav className="flex-1 p-2 space-y-1">
                {navItems.map((item) => (
                    <NavLink
                        key={item.name}
                        to={item.path}
                        className={({ isActive }) =>
                            `flex items-center p-3 text-sm font-medium rounded-lg transition-colors duration-150 ${isActive
                                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300'
                                : 'text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                            }`
                        }
                    >
                        <item.icon className="w-5 h-5 mr-3" />
                        <span className="flex-1">{item.name}</span>
                        <StatusIcon status={item.status} />
                    </NavLink>
                ))}
            </nav>
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 text-xs text-center text-gray-400">
                App Version 1.0.0
            </div>
        </div>
    );
};

export default Sidebar;