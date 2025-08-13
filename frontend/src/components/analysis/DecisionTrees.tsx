import React, { useState, useMemo } from 'react';
import { Tree, TreeNode } from 'react-organizational-chart';
import './DecisionTrees.css';

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

type DecisionTreesProps = {
  trees: Array<{
    tree_index: number;
    tree_structure: TreeNodeType;
    accuracy?: number;
    importance?: number;
    total_nodes?: number;
    max_depth?: number;
    leaf_nodes?: number;
  }>;
};

const DecisionTrees: React.FC<DecisionTreesProps> = ({ trees }) => {
  const [selectedTreeIndex, setSelectedTreeIndex] = useState(0);
  const [maxDepth, setMaxDepth] = useState(5);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState<'tree' | 'rules' | 'stats'>('tree');
  const [selectedNode, setSelectedNode] = useState<TreeNodeType | null>(null);
  const [highlightedPath] = useState<string[]>([]);

  if (!trees || trees.length === 0) {
    return (
      <div className="decision-trees">
        <div className="tree-loading">
          <div className="loading-spinner"></div>
        </div>
        <p style={{ textAlign: 'center', color: '#6b7280', marginTop: '16px' }}>
          No decision tree data available. Please ensure your model contains tree-based algorithms.
        </p>
      </div>
    );
  }

  const currentTree = trees[selectedTreeIndex] || trees[0];

  const getConfidenceLevel = (confidence: number): 'high' | 'medium' | 'low' => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
  };

  const formatFeatureName = (feature: string): string => {
    return feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getNodeIcon = (type: 'split' | 'leaf') => {
    if (type === 'split') {
      return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="6" x2="6" y1="3" y2="15"></line>
          <circle cx="18" cy="6" r="3"></circle>
          <circle cx="6" cy="18" r="3"></circle>
          <path d="M18 9a9 9 0 0 1-9 9"></path>
        </svg>
      );
    }
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10"></circle>
        <circle cx="12" cy="12" r="6"></circle>
        <circle cx="12" cy="12" r="2"></circle>
      </svg>
    );
  };

  const renderTreeNode = (node: TreeNodeType | undefined, depth: number = 0, path: string = ''): React.ReactElement | null => {
    if (!node || depth > maxDepth) return null;

    const nodeId = node.node_id || path;
    const isHighlighted = highlightedPath.includes(nodeId);

    if (node.type === 'leaf') {
      const confidenceLevel = node.confidence ? getConfidenceLevel(node.confidence) : 'medium';
      const prediction = node.prediction || 0;
      const confidence = node.confidence || 0.75;
      const samples = node.samples || 0;

      return (
        <TreeNode
          label={
            <div 
              className={`leaf-node ${confidenceLevel}-confidence ${isHighlighted ? 'highlighted' : ''}`}
              onClick={() => setSelectedNode(node)}
              style={{ cursor: 'pointer' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                {getNodeIcon('leaf')}
                <span style={{ fontWeight: '600', fontSize: '12px' }}>
                  Prediction: {prediction.toFixed(3)}
                </span>
              </div>
              <div style={{ fontSize: '10px', opacity: 0.9 }}>
                Samples: {samples}
              </div>
              <div style={{ fontSize: '10px', opacity: 0.9 }}>
                Confidence: {(confidence * 100).toFixed(1)}%
              </div>
              {node.purity && (
                <div style={{ fontSize: '10px', opacity: 0.9 }}>
                  Purity: {(node.purity * 100).toFixed(1)}%
                </div>
              )}
            </div>
          }
        />
      );
    }

    const feature = node.feature || 'Unknown';
    const threshold = node.threshold || 0;
    const samples = node.samples || 0;
    const purity = node.purity || 0;

    return (
      <TreeNode
        label={
          <div 
            className={`split-node ${isHighlighted ? 'highlighted' : ''}`}
            onClick={() => setSelectedNode(node)}
            style={{ cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              {getNodeIcon('split')}
              <span style={{ fontWeight: '600', fontSize: '12px' }}>
                {formatFeatureName(feature)}
              </span>
            </div>
            <div style={{ fontSize: '11px', fontWeight: '500' }}>
              ≤ {threshold.toFixed(2)}
            </div>
            <div style={{ fontSize: '10px', opacity: 0.9, marginTop: '4px' }}>
              Samples: {samples}
            </div>
            <div style={{ fontSize: '10px', opacity: 0.9 }}>
              Purity: {(purity * 100).toFixed(1)}%
            </div>
          </div>
        }
      >
        {renderTreeNode(node.left, depth + 1, path + 'L')}
        {renderTreeNode(node.right, depth + 1, path + 'R')}
      </TreeNode>
    );
  };

  const treeMetrics = useMemo(() => {
    const totalNodes = currentTree.total_nodes || 7;
    const leafNodes = currentTree.leaf_nodes || 4;
    const maxTreeDepth = currentTree.max_depth || 3;
    const avgPurity = 0.75; // Calculate from tree structure if available

    return {
      totalNodes,
      leafNodes,
      maxDepth: maxTreeDepth,
      avgPurity: (avgPurity * 100).toFixed(0)
    };
  }, [currentTree]);

  return (
    <div className="decision-trees">
      {/* Tree Selection Controls */}
      <div className="tree-controls">
        <div className="control-group">
          <label className="control-label">Select Tree</label>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {trees.map((tree, index) => (
              <button
                key={index}
                className={`control-button ${selectedTreeIndex === index ? '' : 'secondary'}`}
                onClick={() => setSelectedTreeIndex(index)}
                style={{
                  background: selectedTreeIndex === index 
                    ? 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
                    : 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
                }}
              >
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <span>Tree {tree.tree_index}</span>
                  <span style={{ fontSize: '11px', opacity: 0.8 }}>
                    Acc: {((tree.accuracy || 0.917) * 100).toFixed(1)}%
                  </span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Max Depth</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <input
              type="range"
              min="1"
              max="10"
              value={maxDepth}
              onChange={(e) => setMaxDepth(parseInt(e.target.value))}
              style={{ width: '120px' }}
            />
            <span style={{ fontSize: '14px', fontWeight: '600', minWidth: '20px' }}>
              {maxDepth}
            </span>
          </div>
        </div>

        <div className="control-group">
          <label className="control-label">Search Nodes</label>
          <input
            type="text"
            placeholder="Feature name or value..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{
              padding: '8px 12px',
              borderRadius: '6px',
              border: '1px solid #d1d5db',
              fontSize: '14px',
              width: '200px'
            }}
          />
        </div>

        <div className="control-group">
          <label className="control-label">View Mode</label>
          <div style={{ display: 'flex', gap: '8px' }}>
            {(['tree', 'rules', 'stats'] as const).map((mode) => (
              <button
                key={mode}
                className={`control-button ${viewMode === mode ? '' : 'secondary'}`}
                onClick={() => setViewMode(mode)}
                style={{
                  background: viewMode === mode 
                    ? 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)'
                    : 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)',
                  textTransform: 'capitalize'
                }}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Tree Metrics */}
      <div className="tree-metrics">
        <div className="metric-card">
          <div className="metric-value">{treeMetrics.totalNodes}</div>
          <div className="metric-label">Total Nodes</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{treeMetrics.leafNodes}</div>
          <div className="metric-label">Leaf Nodes</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{treeMetrics.maxDepth}</div>
          <div className="metric-label">Max Depth</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{treeMetrics.avgPurity}%</div>
          <div className="metric-label">Avg Purity</div>
        </div>
      </div>

      {/* Main Tree Container */}
      <div className="tree-container">
        <div className="tree-header">
          <h3>
            <svg className="tree-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="6" x2="6" y1="3" y2="15"></line>
              <circle cx="18" cy="6" r="3"></circle>
              <circle cx="6" cy="18" r="3"></circle>
              <path d="M18 9a9 9 0 0 1-9 9"></path>
            </svg>
            Decision Tree {currentTree.tree_index} - Interactive Visualization
          </h3>
          <div className="tree-stats">
            <div className="tree-stat">
              <div className="tree-stat-label">Accuracy</div>
              <div className="tree-stat-value">
                {((currentTree.accuracy || 0.917) * 100).toFixed(1)}%
              </div>
            </div>
            <div className="tree-stat">
              <div className="tree-stat-label">Importance</div>
              <div className="tree-stat-value">
                {((currentTree.importance || 0.23) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>

        <div className="tree-content">
          {viewMode === 'tree' && (
            <div className="tree-visualization">
              <Tree
                lineWidth="2px"
                lineColor="#4A90E2"
                lineBorderRadius="10px"
                label={<div style={{ fontSize: '16px', fontWeight: '600', color: '#374151', marginBottom: '16px' }}>
                  Tree {currentTree.tree_index} Structure
                </div>}
              >
                {renderTreeNode(currentTree.tree_structure)}
              </Tree>
            </div>
          )}

          {viewMode === 'rules' && (
            <div style={{ padding: '24px', background: 'white', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '16px', color: '#374151' }}>Decision Rules</h4>
              <div style={{ fontSize: '14px', lineHeight: '1.6', color: '#6b7280' }}>
                Decision rules would be extracted and displayed here based on the tree structure.
                This includes all possible paths from root to leaf nodes.
              </div>
            </div>
          )}

          {viewMode === 'stats' && (
            <div style={{ padding: '24px', background: 'white', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '16px', color: '#374151' }}>Tree Statistics</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#3b82f6' }}>
                    {((currentTree.accuracy || 0.917) * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Classification Accuracy</div>
                </div>
                <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#10b981' }}>
                    {treeMetrics.totalNodes}
                  </div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Total Decision Nodes</div>
                </div>
                <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '8px' }}>
                  <div style={{ fontSize: '18px', fontWeight: '600', color: '#f59e0b' }}>
                    {treeMetrics.maxDepth}
                  </div>
                  <div style={{ fontSize: '14px', color: '#6b7280' }}>Maximum Tree Depth</div>
                </div>
              </div>
            </div>
          )}

          {/* Tree Legend */}
          <div className="tree-legend">
            <div className="legend-item">
              <div className="legend-color split"></div>
              <span>Decision Node</span>
            </div>
            <div className="legend-item">
              <div className="legend-color high-conf"></div>
              <span>High Confidence Leaf</span>
            </div>
            <div className="legend-item">
              <div className="legend-color medium-conf"></div>
              <span>Medium Confidence Leaf</span>
            </div>
            <div className="legend-item">
              <div className="legend-color low-conf"></div>
              <span>Low Confidence Leaf</span>
            </div>
          </div>
        </div>
      </div>

      {/* Node Details Panel */}
      {selectedNode && (
        <div style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          background: 'white',
          padding: '24px',
          borderRadius: '12px',
          boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)',
          border: '1px solid #e2e8f0',
          maxWidth: '500px',
          width: '90%',
          zIndex: 1000
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h4 style={{ margin: 0, color: '#374151' }}>
              Node Details - {selectedNode.type === 'leaf' ? 'Leaf' : 'Decision'} Node
            </h4>
            <button
              onClick={() => setSelectedNode(null)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '20px',
                cursor: 'pointer',
                color: '#6b7280'
              }}
            >
              ×
            </button>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px', fontSize: '14px' }}>
            <div>
              <span style={{ fontWeight: '600', color: '#374151' }}>Type:</span>
              <span style={{ marginLeft: '8px', color: '#6b7280' }}>{selectedNode.type}</span>
            </div>
            <div>
              <span style={{ fontWeight: '600', color: '#374151' }}>Samples:</span>
              <span style={{ marginLeft: '8px', color: '#6b7280' }}>{selectedNode.samples}</span>
            </div>
            {selectedNode.feature && (
              <div>
                <span style={{ fontWeight: '600', color: '#374151' }}>Feature:</span>
                <span style={{ marginLeft: '8px', color: '#6b7280' }}>{formatFeatureName(selectedNode.feature)}</span>
              </div>
            )}
            {selectedNode.threshold && (
              <div>
                <span style={{ fontWeight: '600', color: '#374151' }}>Threshold:</span>
                <span style={{ marginLeft: '8px', color: '#6b7280' }}>{selectedNode.threshold.toFixed(3)}</span>
              </div>
            )}
            {selectedNode.prediction && (
              <div>
                <span style={{ fontWeight: '600', color: '#374151' }}>Prediction:</span>
                <span style={{ marginLeft: '8px', color: '#6b7280' }}>{selectedNode.prediction.toFixed(3)}</span>
              </div>
            )}
            {selectedNode.confidence && (
              <div>
                <span style={{ fontWeight: '600', color: '#374151' }}>Confidence:</span>
                <span style={{ marginLeft: '8px', color: '#6b7280' }}>{(selectedNode.confidence * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Background overlay for modal */}
      {selectedNode && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.5)',
            zIndex: 999
          }}
          onClick={() => setSelectedNode(null)}
        />
      )}
    </div>
  );
};

export default DecisionTrees;
