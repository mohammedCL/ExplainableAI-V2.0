import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import Sidebar from './Sidebar';

const MainLayout: React.FC = () => {
    return (
        <div className="flex h-screen bg-gray-50 dark:bg-gray-800 text-gray-800 dark:text-gray-200">
            <Sidebar />
            <main className="relative flex-1 overflow-y-auto">
                <div className="sticky top-0 z-20 flex justify-end p-4 bg-transparent">
                    <Link
                        to="/upload"
                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 text-sm font-medium shadow"
                    >
                        Upload Model & Data
                    </Link>
                </div>
                <div className="px-6 pb-6">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default MainLayout;