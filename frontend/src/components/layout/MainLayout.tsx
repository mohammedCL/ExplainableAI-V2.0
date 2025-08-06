import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import Sidebar from './Sidebar';

const MainLayout: React.FC = () => {
    return (
        <div className="flex h-screen bg-gray-50 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
            <Sidebar />
            <main className="flex-1 overflow-y-auto">
                <div className="absolute top-4 right-4 z-10">
                    <Link
                        to="/upload"
                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 text-sm font-medium"
                    >
                        Upload Model & Data
                    </Link>
                </div>
                <Outlet />
            </main>
        </div>
    );
};

export default MainLayout;