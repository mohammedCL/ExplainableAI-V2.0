import React, { useState } from 'react';
import { X, Brain, Lightbulb, CheckCircle } from 'lucide-react';
import { explainWithAI } from '../../services/api';

interface AIExplanation {
  summary: string;
  detailed_explanation: string;
  key_takeaways: string[];
}

interface AIExplanationPanelProps {
  isOpen: boolean;
  onClose: () => void;
  analysisType: string;
  analysisData: any;
  title?: string;
}

const AIExplanationPanel: React.FC<AIExplanationPanelProps> = ({
  isOpen,
  onClose,
  analysisType,
  analysisData,
  title = "Explain with AI"
}) => {
  const [explanation, setExplanation] = useState<AIExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateExplanation = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await explainWithAI(analysisType, analysisData);
      setExplanation(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate AI explanation');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black bg-opacity-50 transition-opacity"
        onClick={onClose}
      />
      
      {/* Panel */}
      <div className="absolute right-0 top-0 h-full w-full max-w-md bg-white dark:bg-gray-800 shadow-xl transform transition-transform duration-300 ease-in-out">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex items-center space-x-2">
              <Brain className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                {title}
              </h2>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!explanation && !loading && !error && (
              <div className="text-center py-8">
                <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Get an AI-powered explanation of your analysis results
                </p>
                <button
                  onClick={generateExplanation}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center mx-auto space-x-2"
                >
                  <Brain className="w-4 h-4" />
                  <span>Generate Explanation</span>
                </button>
              </div>
            )}

            {loading && (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600 dark:text-gray-400">
                  AI is analyzing your data...
                </p>
              </div>
            )}

            {error && (
              <div className="text-center py-8">
                <div className="w-12 h-12 bg-red-100 dark:bg-red-900/50 rounded-full flex items-center justify-center mx-auto mb-4">
                  <X className="w-6 h-6 text-red-600" />
                </div>
                <p className="text-red-600 dark:text-red-400 mb-4">
                  {error}
                </p>
                <button
                  onClick={generateExplanation}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  Try Again
                </button>
              </div>
            )}

            {explanation && (
              <div className="space-y-6">
                {/* Summary */}
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2 flex items-center">
                    <Lightbulb className="w-4 h-4 mr-2" />
                    Summary
                  </h3>
                  <p className="text-blue-800 dark:text-blue-200 text-sm">
                    {explanation.summary}
                  </p>
                </div>

                {/* Detailed Explanation */}
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                    Detailed Explanation
                  </h3>
                  <p className="text-gray-700 dark:text-gray-300 text-sm leading-relaxed">
                    {explanation.detailed_explanation}
                  </p>
                </div>

                {/* Key Takeaways */}
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
                    Key Takeaways
                  </h3>
                  <ul className="space-y-2">
                    {explanation.key_takeaways.map((takeaway, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                        <span className="text-gray-700 dark:text-gray-300 text-sm">
                          {takeaway}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Regenerate Button */}
                <button
                  onClick={generateExplanation}
                  className="w-full bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 px-4 py-2 rounded-lg transition-colors flex items-center justify-center space-x-2"
                >
                  <Brain className="w-4 h-4" />
                  <span>Regenerate Explanation</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIExplanationPanel;
