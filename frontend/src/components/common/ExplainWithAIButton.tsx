import React from 'react';
import { Brain } from 'lucide-react';

interface ExplainWithAIButtonProps {
  onClick: () => void;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

const ExplainWithAIButton: React.FC<ExplainWithAIButtonProps> = ({
  onClick,
  className = '',
  size = 'md'
}) => {
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base'
  };

  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center space-x-2 
        bg-gradient-to-r from-blue-600 to-blue-700 
        hover:from-blue-700 hover:to-blue-800 
        text-white font-medium rounded-lg 
        transition-all duration-200 
        shadow-md hover:shadow-lg 
        transform hover:scale-105 
        ${sizeClasses[size]} 
        ${className}
      `}
      title="Explain with AI"
    >
      <Brain className={`${size === 'sm' ? 'w-4 h-4' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'}`} />
      <span>Explain with AI</span>
    </button>
  );
};

export default ExplainWithAIButton;
