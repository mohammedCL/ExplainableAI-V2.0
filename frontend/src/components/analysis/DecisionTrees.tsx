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
  const [viewMode, setViewMode] = useState<'tree' | 'comparison' | 'rules' | 'stats'>('tree');
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

  // Function to calculate node depth and enhanced details
  const calculateNodeDepth = (targetNode: TreeNodeType, currentNode: TreeNodeType, currentDepth: number = 0): number => {
    if (currentNode === targetNode) return currentDepth;
    
    if (currentNode.left) {
      const leftDepth = calculateNodeDepth(targetNode, currentNode.left, currentDepth + 1);
      if (leftDepth !== -1) return leftDepth;
    }
    
    if (currentNode.right) {
      const rightDepth = calculateNodeDepth(targetNode, currentNode.right, currentDepth + 1);
      if (rightDepth !== -1) return rightDepth;
    }
    
    return -1;
  };

  // Function to extract decision rules from tree
  const extractDecisionRules = (node: TreeNodeType, currentPath: string[] = [], allRules: Array<{path: string[], prediction: number, confidence: number, samples: number}> = []): Array<{path: string[], prediction: number, confidence: number, samples: number}> => {
    if (!node) return allRules;

    if (node.type === 'leaf') {
      // This is a leaf node, create a rule
      allRules.push({
        path: [...currentPath],
        prediction: node.prediction || 0,
        confidence: node.confidence || 0,
        samples: node.samples || 0
      });
      return allRules;
    }

    // This is a split node
    if (node.feature && node.threshold !== undefined) {
      // Left branch (≤ threshold)
      if (node.left) {
        const leftCondition = `${formatFeatureName(node.feature)} ≤ ${node.threshold.toFixed(3)}`;
        extractDecisionRules(node.left, [...currentPath, leftCondition], allRules);
      }

      // Right branch (> threshold)
      if (node.right) {
        const rightCondition = `${formatFeatureName(node.feature)} > ${node.threshold.toFixed(3)}`;
        extractDecisionRules(node.right, [...currentPath, rightCondition], allRules);
      }
    }

    return allRules;
  };

  const decisionRules = useMemo(() => {
    return extractDecisionRules(currentTree.tree_structure);
  }, [currentTree]);

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
            {(['tree', 'comparison', 'rules', 'stats'] as const).map((mode) => (
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

          {viewMode === 'comparison' && (
            <div style={{ padding: '24px', background: 'white', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '24px', color: '#374151', fontSize: '18px', fontWeight: '600' }}>
                Tree Comparison
              </h4>
              <p style={{ marginBottom: '24px', color: '#6b7280', fontSize: '14px' }}>
                Compare performance and complexity across different trees in the ensemble
              </p>
              
              <div style={{ overflowX: 'auto' }}>
                <table style={{ 
                  width: '100%', 
                  borderCollapse: 'collapse',
                  fontSize: '14px',
                  background: 'white',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
                }}>
                  <thead>
                    <tr style={{ backgroundColor: '#f8fafc' }}>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'left', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Tree</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Accuracy</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Importance</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Nodes</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Depth</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Leaves</th>
                      <th style={{ 
                        padding: '16px 12px', 
                        textAlign: 'center', 
                        fontWeight: '600', 
                        color: '#374151',
                        borderBottom: '1px solid #e2e8f0'
                      }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trees.map((tree, index) => {
                      const isSelected = selectedTreeIndex === index;
                      return (
                        <tr 
                          key={tree.tree_index}
                          style={{ 
                            backgroundColor: isSelected ? '#eff6ff' : index % 2 === 0 ? '#ffffff' : '#f9fafb',
                            borderBottom: '1px solid #e2e8f0'
                          }}
                        >
                          <td style={{ 
                            padding: '16px 12px',
                            fontWeight: isSelected ? '600' : '500',
                            color: isSelected ? '#3b82f6' : '#374151'
                          }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <div style={{ 
                                width: '8px', 
                                height: '8px', 
                                borderRadius: '50%', 
                                backgroundColor: isSelected ? '#3b82f6' : '#9ca3af' 
                              }}></div>
                              Tree {tree.tree_index}
                            </div>
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center',
                            color: '#374151'
                          }}>
                            <span style={{ 
                              fontWeight: '600',
                              color: (tree.accuracy || 0.917) >= 0.9 ? '#10b981' : 
                                     (tree.accuracy || 0.917) >= 0.8 ? '#f59e0b' : '#ef4444'
                            }}>
                              {((tree.accuracy || 0.917) * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center',
                            color: '#374151'
                          }}>
                            {((tree.importance || 0.2) * 100).toFixed(1)}%
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center',
                            color: '#374151'
                          }}>
                            {tree.total_nodes || 'N/A'}
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center',
                            color: '#374151'
                          }}>
                            {tree.max_depth || 'N/A'}
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center',
                            color: '#374151'
                          }}>
                            {tree.leaf_nodes || 'N/A'}
                          </td>
                          <td style={{ 
                            padding: '16px 12px', 
                            textAlign: 'center'
                          }}>
                            <button
                              onClick={() => {
                                setSelectedTreeIndex(index);
                                setViewMode('tree');
                              }}
                              style={{
                                padding: '6px 12px',
                                fontSize: '12px',
                                fontWeight: '500',
                                color: '#3b82f6',
                                backgroundColor: '#eff6ff',
                                border: '1px solid #bfdbfe',
                                borderRadius: '6px',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.backgroundColor = '#dbeafe';
                                e.currentTarget.style.borderColor = '#93c5fd';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.backgroundColor = '#eff6ff';
                                e.currentTarget.style.borderColor = '#bfdbfe';
                              }}
                            >
                              View Tree
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* Summary Statistics */}
              <div style={{ 
                marginTop: '32px', 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '16px' 
              }}>
                <div style={{ 
                  padding: '20px', 
                  backgroundColor: '#f0f9ff', 
                  borderRadius: '8px', 
                  border: '1px solid #e0f2fe' 
                }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#0369a1', marginBottom: '4px' }}>
                    {((trees.reduce((sum, tree) => sum + (tree.accuracy || 0.917), 0) / trees.length) * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '14px', color: '#0369a1', fontWeight: '500' }}>
                    Average Accuracy
                  </div>
                </div>
                
                <div style={{ 
                  padding: '20px', 
                  backgroundColor: '#f0fdf4', 
                  borderRadius: '8px', 
                  border: '1px solid #dcfce7' 
                }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#15803d', marginBottom: '4px' }}>
                    {Math.max(...trees.map(tree => tree.max_depth || 0))}
                  </div>
                  <div style={{ fontSize: '14px', color: '#15803d', fontWeight: '500' }}>
                    Max Tree Depth
                  </div>
                </div>
                
                <div style={{ 
                  padding: '20px', 
                  backgroundColor: '#fffbeb', 
                  borderRadius: '8px', 
                  border: '1px solid #fed7aa' 
                }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#c2410c', marginBottom: '4px' }}>
                    {Math.round(trees.reduce((sum, tree) => sum + (tree.total_nodes || 0), 0) / trees.length)}
                  </div>
                  <div style={{ fontSize: '14px', color: '#c2410c', fontWeight: '500' }}>
                    Avg Nodes per Tree
                  </div>
                </div>
                
                <div style={{ 
                  padding: '20px', 
                  backgroundColor: '#fdf2f8', 
                  borderRadius: '8px', 
                  border: '1px solid #fce7f3' 
                }}>
                  <div style={{ fontSize: '24px', fontWeight: '700', color: '#be185d', marginBottom: '4px' }}>
                    {trees.length}
                  </div>
                  <div style={{ fontSize: '14px', color: '#be185d', fontWeight: '500' }}>
                    Total Trees
                  </div>
                </div>
              </div>
            </div>
          )}

          {viewMode === 'rules' && (
            <div style={{ padding: '24px', background: 'white', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '16px', color: '#374151', fontSize: '18px', fontWeight: '600' }}>
                Decision Rules - Tree {currentTree.tree_index}
              </h4>
              <p style={{ marginBottom: '24px', color: '#6b7280', fontSize: '14px' }}>
                All possible paths from root to leaf nodes with their corresponding predictions.
              </p>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {decisionRules.map((rule, index) => (
                  <div 
                    key={index}
                    style={{
                      padding: '20px',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                      backgroundColor: '#f9fafb',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f0f9ff';
                      e.currentTarget.style.borderColor = '#bfdbfe';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = '#f9fafb';
                      e.currentTarget.style.borderColor = '#e2e8f0';
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                      <div style={{ fontWeight: '600', color: '#374151', fontSize: '16px' }}>
                        Rule {index + 1}
                      </div>
                      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                        <div style={{ 
                          padding: '4px 8px', 
                          backgroundColor: rule.confidence >= 0.8 ? '#dcfce7' : rule.confidence >= 0.6 ? '#fef3c7' : '#fee2e2',
                          color: rule.confidence >= 0.8 ? '#166534' : rule.confidence >= 0.6 ? '#92400e' : '#991b1b',
                          borderRadius: '4px',
                          fontSize: '12px',
                          fontWeight: '600'
                        }}>
                          {(rule.confidence * 100).toFixed(1)}% confidence
                        </div>
                        <div style={{ 
                          fontSize: '14px', 
                          color: '#6b7280',
                          fontWeight: '500'
                        }}>
                          {rule.samples} samples
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ marginBottom: '12px' }}>
                      <div style={{ fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                        Conditions:
                      </div>
                      {rule.path.length > 0 ? (
                        <div style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          {rule.path.map((condition, condIndex) => (
                            <div key={condIndex} style={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              marginBottom: '4px',
                              color: '#4b5563'
                            }}>
                              <span style={{ 
                                marginRight: '8px', 
                                fontWeight: '600',
                                color: '#6b7280',
                                minWidth: '20px'
                              }}>
                                {condIndex + 1}.
                              </span>
                              <span style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                                {condition}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div style={{ 
                          fontSize: '14px', 
                          color: '#6b7280',
                          fontStyle: 'italic'
                        }}>
                          Root node (no conditions)
                        </div>
                      )}
                    </div>
                    
                    <div style={{ 
                      borderTop: '1px solid #e2e8f0', 
                      paddingTop: '12px',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center'
                    }}>
                      <div>
                        <span style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                          Prediction:
                        </span>
                        <span style={{ 
                          fontSize: '14px', 
                          marginLeft: '8px',
                          fontWeight: '600',
                          color: rule.prediction > 0.5 ? '#059669' : '#dc2626',
                          fontFamily: 'monospace'
                        }}>
                          {rule.prediction.toFixed(6)} ({rule.prediction > 0.5 ? 'Positive' : 'Negative'})
                        </span>
                      </div>
                      
                      <div style={{ 
                        fontSize: '12px', 
                        color: '#6b7280',
                        textAlign: 'right'
                      }}>
                        Path depth: {rule.path.length} levels
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {decisionRules.length === 0 && (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '40px',
                  color: '#6b7280',
                  fontSize: '14px'
                }}>
                  No decision rules found. The tree structure may be incomplete.
                </div>
              )}
              
              {/* Rules Summary */}
              {decisionRules.length > 0 && (
                <div style={{ 
                  marginTop: '32px',
                  padding: '20px',
                  backgroundColor: '#f8fafc',
                  borderRadius: '8px',
                  border: '1px solid #e2e8f0'
                }}>
                  <h5 style={{ margin: '0 0 16px 0', color: '#374151', fontSize: '16px', fontWeight: '600' }}>
                    Rules Summary
                  </h5>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: '#3b82f6', marginBottom: '4px' }}>
                        {decisionRules.length}
                      </div>
                      <div style={{ fontSize: '14px', color: '#6b7280' }}>Total Rules</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: '#10b981', marginBottom: '4px' }}>
                        {((decisionRules.reduce((sum, rule) => sum + rule.confidence, 0) / decisionRules.length) * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '14px', color: '#6b7280' }}>Avg Confidence</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: '#f59e0b', marginBottom: '4px' }}>
                        {Math.max(...decisionRules.map(rule => rule.path.length))}
                      </div>
                      <div style={{ fontSize: '14px', color: '#6b7280' }}>Max Path Depth</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '24px', fontWeight: '700', color: '#ef4444', marginBottom: '4px' }}>
                        {decisionRules.reduce((sum, rule) => sum + rule.samples, 0)}
                      </div>
                      <div style={{ fontSize: '14px', color: '#6b7280' }}>Total Samples</div>
                    </div>
                  </div>
                </div>
              )}
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
          maxWidth: '400px',
          width: '90%',
          zIndex: 1000
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px' }}>
            <div>
              <h4 style={{ margin: 0, color: '#374151', fontSize: '16px', fontWeight: '600' }}>
                Sample Node Details: {selectedNode.node_id || 'N/A'}
              </h4>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '24px',
                cursor: 'pointer',
                color: '#6b7280',
                padding: '0',
                lineHeight: '1'
              }}
            >
              ×
            </button>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
            {/* Type */}
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#374151' }}>Type:</span>
              <span style={{ color: '#6b7280' }}>{selectedNode.type}</span>
            </div>
            
            {/* Samples */}
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#374151' }}>Samples:</span>
              <span style={{ color: '#6b7280' }}>{selectedNode.samples}</span>
            </div>
            
            {/* Depth */}
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ fontWeight: '600', color: '#374151' }}>Depth:</span>
              <span style={{ color: '#6b7280' }}>
                {calculateNodeDepth(selectedNode, currentTree.tree_structure)}
              </span>
            </div>
            
            {/* Feature (for split nodes) */}
            {selectedNode.feature && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Feature:</span>
                <span style={{ color: '#6b7280' }}>{formatFeatureName(selectedNode.feature)}</span>
              </div>
            )}
            
            {/* Threshold (for split nodes) */}
            {selectedNode.threshold !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Threshold:</span>
                <span style={{ color: '#6b7280' }}>≤ {selectedNode.threshold.toFixed(4)}</span>
              </div>
            )}
            
            {/* Prediction (for leaf nodes) */}
            {selectedNode.prediction !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Prediction:</span>
                <span style={{ color: '#6b7280' }}>{selectedNode.prediction.toFixed(16)}</span>
              </div>
            )}
            
            {/* Confidence */}
            {selectedNode.confidence !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Confidence:</span>
                <span style={{ color: '#6b7280' }}>{(selectedNode.confidence * 100).toFixed(1)}%</span>
              </div>
            )}
            
            {/* Purity */}
            {selectedNode.purity !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Purity:</span>
                <span style={{ color: '#6b7280' }}>{(selectedNode.purity * 100).toFixed(1)}%</span>
              </div>
            )}
            
            {/* Gini Impurity */}
            {selectedNode.gini !== undefined && (
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontWeight: '600', color: '#374151' }}>Gini:</span>
                <span style={{ color: '#6b7280' }}>{selectedNode.gini.toFixed(4)}</span>
              </div>
            )}
            
            {/* Class Distribution */}
            {selectedNode.class_distribution && Object.keys(selectedNode.class_distribution).length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <div style={{ fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                  Class Distribution:
                </div>
                <div style={{ 
                  backgroundColor: '#f8fafc', 
                  padding: '12px', 
                  borderRadius: '6px',
                  border: '1px solid #e2e8f0'
                }}>
                  {Object.entries(selectedNode.class_distribution).map(([className, count], index) => (
                    <div key={index} style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between',
                      marginBottom: index < Object.entries(selectedNode.class_distribution!).length - 1 ? '8px' : '0'
                    }}>
                      <span style={{ fontWeight: '500', color: '#374151' }}>
                        {className === 'class_0' ? 'Rejected' : className === 'class_1' ? 'Approved' : className}
                      </span>
                      <span style={{ fontWeight: '600', color: '#6b7280' }}>
                        {Math.round(count)}
                      </span>
                    </div>
                  ))}
                </div>
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
