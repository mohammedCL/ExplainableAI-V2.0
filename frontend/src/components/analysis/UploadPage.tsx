import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadModelAndData, uploadModelAndSeparateDatasets } from '../../services/api';
import { UploadCloud, Loader2, Database, Split } from 'lucide-react';

const UploadPage: React.FC = () => {
    const [uploadMode, setUploadMode] = useState<'single' | 'separate'>('single');
    const [modelFile, setModelFile] = useState<File | null>(null);
    const [dataFile, setDataFile] = useState<File | null>(null);
    const [trainFile, setTrainFile] = useState<File | null>(null);
    const [testFile, setTestFile] = useState<File | null>(null);
    const [targetColumn, setTargetColumn] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        
        if (uploadMode === 'single') {
            if (!modelFile || !dataFile || !targetColumn) {
                setError('Please provide all fields.');
                return;
            }
        } else {
            if (!modelFile || !trainFile || !testFile || !targetColumn) {
                setError('Please provide all fields.');
                return;
            }
        }
        
        setError('');
        setIsLoading(true);

        try {
            if (uploadMode === 'single') {
                await uploadModelAndData(modelFile!, dataFile!, targetColumn);
            } else {
                await uploadModelAndSeparateDatasets(modelFile!, trainFile!, testFile!, targetColumn);
            }
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
            <div className="w-full max-w-lg p-8 space-y-6 bg-white rounded-lg shadow-md dark:bg-gray-800">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Start New Analysis</h1>
                    <p className="text-gray-600 dark:text-gray-400">Upload your model and dataset to begin.</p>
                </div>

                {/* Upload Mode Selection */}
                <div className="space-y-3">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Upload Mode</label>
                    <div className="flex space-x-4">
                        <button
                            type="button"
                            onClick={() => setUploadMode('single')}
                            className={`flex-1 p-3 border rounded-lg text-sm font-medium transition-colors ${
                                uploadMode === 'single'
                                    ? 'border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                                    : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                            }`}
                        >
                            <Database className="w-4 h-4 mx-auto mb-1" />
                            Single Dataset
                            <div className="text-xs text-gray-500 dark:text-gray-400">Auto train/test split</div>
                        </button>
                        <button
                            type="button"
                            onClick={() => setUploadMode('separate')}
                            className={`flex-1 p-3 border rounded-lg text-sm font-medium transition-colors ${
                                uploadMode === 'separate'
                                    ? 'border-blue-500 bg-blue-50 text-blue-700 dark:bg-blue-900 dark:text-blue-200'
                                    : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                            }`}
                        >
                            <Split className="w-4 h-4 mx-auto mb-1" />
                            Separate Datasets
                            <div className="text-xs text-gray-500 dark:text-gray-400">Pre-split datasets</div>
                        </button>
                    </div>
                </div>

                <form className="space-y-6" onSubmit={handleSubmit}>
                    <div>
                        <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                            Model File
                            <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">(.joblib, .pkl, .pickle, .onnx)</span>
                        </label>
                        <input 
                            type="file" 
                            accept=".joblib,.pkl,.pickle,.onnx" 
                            onChange={e => setModelFile(e.target.files?.[0] || null)} 
                            className="file-input" 
                        />
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                            Supports scikit-learn models (.joblib, .pkl, .pickle) and ONNX models (.onnx)
                        </p>
                    </div>

                    {uploadMode === 'single' ? (
                        <div>
                            <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Dataset File (.csv)</label>
                            <input type="file" accept=".csv" onChange={e => setDataFile(e.target.files?.[0] || null)} className="file-input" />
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                Will be automatically split into 80% training and 20% testing
                            </p>
                        </div>
                    ) : (
                        <>
                            <div>
                                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Training Dataset (.csv)</label>
                                <input type="file" accept=".csv" onChange={e => setTrainFile(e.target.files?.[0] || null)} className="file-input" />
                            </div>
                            <div>
                                <label className="block mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">Testing Dataset (.csv)</label>
                                <input type="file" accept=".csv" onChange={e => setTestFile(e.target.files?.[0] || null)} className="file-input" />
                            </div>
                        </>
                    )}

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