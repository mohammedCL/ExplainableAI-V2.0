import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadModelAndData } from '../../services/api';
import { UploadCloud, Loader2 } from 'lucide-react';

const UploadPage: React.FC = () => {
    const [modelFile, setModelFile] = useState<File | null>(null);
    const [dataFile, setDataFile] = useState<File | null>(null);
    const [targetColumn, setTargetColumn] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!modelFile || !dataFile || !targetColumn) {
            setError('Please provide all fields.');
            return;
        }
        setError('');
        setIsLoading(true);

        try {
            await uploadModelAndData(modelFile, dataFile, targetColumn);
            // On success, navigate to the overview page
            navigate('/overview');
        } catch (err: any) {
            setError(err.response?.data?.detail || 'An unexpected error occurred.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100 dark:bg-gray-900">
            <div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Start New Analysis</h1>
                    <p className="text-gray-600 dark:text-gray-400">Upload your model and dataset to begin.</p>
                </div>
                <form className="space-y-6" onSubmit={handleSubmit}>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Model File (.joblib)</label>
                        <input type="file" accept=".joblib" onChange={e => setModelFile(e.target.files?.[0] || null)} className="file-input" />
                    </div>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Dataset File (.csv)</label>
                        <input type="file" accept=".csv" onChange={e => setDataFile(e.target.files?.[0] || null)} className="file-input" />
                    </div>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Target Column Name</label>
                        <input type="text" value={targetColumn} onChange={e => setTargetColumn(e.target.value)} placeholder="e.g., 'target' or 'has_churned'" className="w-full px-3 py-2 border rounded-md dark:bg-gray-700 dark:border-gray-600" />
                    </div>
                    {error && <p className="text-sm text-red-500">{error}</p>}
                    <button type="submit" disabled={isLoading} className="w-full px-4 py-2 font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-blue-300 flex items-center justify-center">
                        {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <UploadCloud className="w-5 h-5 mr-2" />}
                        {isLoading ? 'Analyzing...' : 'Upload & Analyze'}
                    </button>
                </form>
            </div>
            <style>{`.file-input { display: block; width: 100%; font-size: 0.875rem; color: #4b5563; file:px-4 file:py-2 file:border-0 file:rounded-md file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100; }`}</style>
        </div>
    );
};

export default UploadPage;