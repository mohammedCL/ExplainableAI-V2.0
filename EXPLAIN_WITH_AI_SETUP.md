# Explain with AI Feature Setup Guide

## Overview
The "Explain with AI" feature provides AI-powered explanations of data analysis results, making insights accessible to users without technical backgrounds. It integrates with AWS Bedrock to generate clear, structured explanations.

## Features
- **AI-Powered Explanations**: Uses Claude 3 Sonnet via AWS Bedrock
- **Structured Output**: Provides summary, detailed explanation, and key takeaways
- **Fallback Mode**: Gracefully handles service unavailability
- **Responsive UI**: Clean right-side panel with smooth animations
- **Reusable Components**: Easy to integrate into any analysis tab

## Backend Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. AWS Configuration
The feature requires AWS Bedrock access. Configure your AWS credentials using one of these methods:

#### Option A: Environment Variables
```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_REGION="us-east-1"  # or your preferred region
```

#### Option B: AWS Credentials File
```bash
# ~/.aws/credentials
[default]
aws_access_key_id = your_access_key
aws_secret_access_key = your_secret_key

# ~/.aws/config
[default]
region = us-east-1
```

#### Option C: IAM Role (for EC2/ECS)
If running on AWS infrastructure, attach an IAM role with Bedrock permissions.

### 3. Required AWS Permissions
Your AWS user/role needs the following permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        }
    ]
}
```

### 4. Start the Backend
```bash
cd backend
uvicorn app.main:app --reload
```

## Frontend Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start the Frontend
```bash
npm run dev
```

## Usage

### 1. Basic Integration
The feature is already integrated into the ModelOverview component. To add it to other components:

```tsx
import ExplainWithAIButton from '../common/ExplainWithAIButton';
import AIExplanationPanel from '../common/AIExplanationPanel';

const YourComponent = () => {
    const [showAIExplanation, setShowAIExplanation] = useState(false);
    
    return (
        <div>
            {/* Header with Explain with AI button */}
            <div className="flex justify-between items-center">
                <h1>Your Analysis</h1>
                <ExplainWithAIButton 
                    onClick={() => setShowAIExplanation(true)}
                    size="md"
                />
            </div>
            
            {/* AI Explanation Panel */}
            <AIExplanationPanel
                isOpen={showAIExplanation}
                onClose={() => setShowAIExplanation(false)}
                analysisType="your_analysis_type"
                analysisData={yourData}
                title="Your Analysis - AI Explanation"
            />
            
            {/* Your existing content */}
        </div>
    );
};
```

### 2. Button Placement
Place the "Explain with AI" button in the top-right of each feature tab for consistency.

### 3. Analysis Types
Use descriptive analysis types for better AI explanations:
- `overview` - Model overview and performance metrics
- `feature_importance` - Feature importance analysis
- `classification_stats` - Classification statistics
- `decision_tree` - Decision tree visualization
- `feature_dependence` - Feature dependence analysis
- `feature_interactions` - Feature interaction analysis

## API Endpoint

### POST /analysis/explain-with-ai
Generates AI explanations for analysis data.

**Request Body:**
```json
{
    "analysis_type": "overview",
    "analysis_data": {
        "performance_metrics": {...},
        "metadata": {...}
    }
}
```

**Response:**
```json
{
    "summary": "Brief overview of key insights",
    "detailed_explanation": "Step-by-step breakdown...",
    "key_takeaways": [
        "Key takeaway 1",
        "Key takeaway 2",
        "Key takeaway 3"
    ]
}
```

## Customization

### 1. LLM Model
Change the model in `ai_explanation_service.py`:
```python
self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # Faster, cheaper
# or
self.model_id = "anthropic.claude-3-opus-20240229-v1:0"   # Most capable
```

### 2. Prompt Engineering
Modify the system prompt in `_create_prompt()` method to:
- Change the explanation style
- Add domain-specific instructions
- Modify output format requirements

### 3. UI Styling
Customize the panel appearance by modifying:
- `AIExplanationPanel.tsx` - Panel layout and styling
- `ExplainWithAIButton.tsx` - Button appearance

## Troubleshooting

### 1. AWS Credentials Issues
- Verify credentials are correctly set
- Check IAM permissions for Bedrock
- Ensure region supports Bedrock service

### 2. Service Unavailable
- Check AWS service status
- Verify network connectivity
- Review CloudWatch logs for errors

### 3. Fallback Mode
If AI service is unavailable, the feature gracefully falls back to basic explanations.

## Security Considerations

1. **API Authentication**: All endpoints require valid tokens
2. **Data Privacy**: Analysis data is sent to AWS Bedrock for processing
3. **Rate Limiting**: Consider implementing rate limiting for AI explanation requests
4. **Logging**: Monitor usage and costs through AWS CloudWatch

## Cost Optimization

1. **Model Selection**: Use Claude Haiku for development/testing
2. **Caching**: Implement explanation caching to avoid repeated API calls
3. **Rate Limiting**: Prevent excessive usage
4. **Monitoring**: Set up CloudWatch alarms for cost thresholds

## Future Enhancements

1. **Explanation Caching**: Store explanations to reduce API calls
2. **Multi-language Support**: Generate explanations in different languages
3. **Custom Prompts**: Allow users to customize explanation focus
4. **Batch Processing**: Generate explanations for multiple analyses at once
5. **Feedback Loop**: Collect user feedback to improve explanations
