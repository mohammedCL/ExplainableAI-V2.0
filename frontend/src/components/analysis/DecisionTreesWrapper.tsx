import React, { useState, useEffect } from 'react';
import DecisionTrees from './DecisionTrees';
import { getDecisionTree } from '../../services/api';
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

type TreeNodeType = {
  type: 'split' | 'leaf';
  feature?: string;
  threshold?: number;
  samples: number;
  prediction?: number;
  confidence?: number;
  purity?: number;
  gini?: number;
  left?: TreeNodeType;
  right?: TreeNodeType;
  node_id?: string;
  class_distribution?: { [key: string]: number };
};

interface TreeData {
  tree_index: number;
  accuracy: number;
  importance: number;
  total_nodes: number;
  leaf_nodes: number;
  max_depth: number;
  tree_structure: TreeNodeType;
}

const DecisionTreesWrapper: React.FC = () => {
  const [treesData, setTreesData] = useState<TreeData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showAIExplanation, setShowAIExplanation] = useState(false);

  useEffect(() => {
    const fetchTreesData = async () => {
      try {
        console.log('Fetching decision tree data from API...');
        const response = await getDecisionTree();
        console.log('Decision tree response:', response);
        console.log('Number of trees received:', response.trees?.length || 0);
        if (response.trees && response.trees.length > 0) {
          console.log('First tree sample:', {
            tree_index: response.trees[0].tree_index,
            accuracy: response.trees[0].accuracy,
            total_nodes: response.trees[0].total_nodes,
            max_depth: response.trees[0].max_depth,
            root_feature: response.trees[0].tree_structure?.feature
          });
        }
        setTreesData(response.trees || []);
        setError(null);
      } catch (err: any) {
        console.error('Error fetching decision tree data:', err);
        const errorMessage = err.response?.data?.detail || 'Failed to fetch decision tree data.';
        setError(errorMessage);

        // Only use mock data if it's a connection error or if the backend is not available
        if (errorMessage.includes('Failed to fetch') || errorMessage.includes('Network Error')) {
          console.log('Using mock data due to API connection error...');
          // Mock data for testing when backend is not available
          const mockData: TreeData[] = [
            {
              tree_index: 0,
              accuracy: 0.917,
              importance: 0.23,
              total_nodes: 7,
              leaf_nodes: 4,
              max_depth: 3,
              tree_structure: {
                type: 'split',
                feature: 'mean_radius',
                threshold: 13.4,
                samples: 100,
                purity: 0.8,
                gini: 0.2,
                node_id: 'node_0',
                left: {
                  type: 'leaf',
                  samples: 45,
                  prediction: 0,
                  confidence: 0.95,
                  purity: 1.0,
                  gini: 0.0,
                  node_id: 'node_1',
                  class_distribution: { class_0: 42, class_1: 3 }
                },
                right: {
                  type: 'split',
                  feature: 'mean_texture',
                  threshold: 20.1,
                  samples: 55,
                  purity: 0.7,
                  gini: 0.3,
                  node_id: 'node_2',
                  left: {
                    type: 'leaf',
                    samples: 25,
                    prediction: 1,
                    confidence: 0.88,
                    purity: 1.0,
                    gini: 0.0,
                    node_id: 'node_3',
                    class_distribution: { class_0: 3, class_1: 22 }
                  },
                  right: {
                    type: 'leaf',
                    samples: 30,
                    prediction: 1,
                    confidence: 0.93,
                    purity: 1.0,
                    gini: 0.0,
                    node_id: 'node_4',
                    class_distribution: { class_0: 2, class_1: 28 }
                  }
                }
              }
            }
          ];
          setTreesData(mockData);
          setError('Using mock data - backend not available');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchTreesData();
  }, []);

  if (loading) {
    return (
      <div className="p-6 flex justify-center items-center h-full">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-gray-600">Loading decision tree data...</p>
        </div>
      </div>
    );
  }

  if (error && treesData.length === 0) {
    return (
      <div className="p-6 text-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-red-800 mb-2">Error Loading Decision Trees</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <p className="text-sm text-gray-600">
            Make sure the backend is running and the model is loaded.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Decision Trees</h1>
        <ExplainWithAIButton onClick={() => setShowAIExplanation(true)} size="md" />
      </div>

      <DecisionTrees trees={treesData} />

      <AIExplanationPanel
        isOpen={showAIExplanation}
        onClose={() => setShowAIExplanation(false)}
        analysisType="decision_tree"
        analysisData={{
          trees: treesData
        }}
        title="Decision Trees - AI Explanation"
      />
    </div>
  );
};

export default DecisionTreesWrapper;
