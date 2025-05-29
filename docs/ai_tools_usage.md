# AI Tools Usage Guide

## Overview
This document outlines how the Financial Assistant leverages various AI tools and Language Learning Models (LLMs) to provide intelligent financial analysis and insights.

## 🤖 AI Architecture

### Primary LLM Providers
1. **Groq (Primary)** - Fast inference with Llama 3.1 70B
2. **Google Gemini (Backup)** - Reliable fallback with Flash model
3. **Intelligent Fallbacks** - Context-aware responses when APIs fail

### Provider Priority Order
```
Groq → Gemini → Intelligent Fallback
```

## 🔧 LLM Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional (backup)
GEMINI_API_KEY=your_gemini_api_key_here
```

### Model Selection
- **Groq**: `llama-3.1-70b-versatile` (Primary choice for speed and quality)
- **Gemini**: `gemini-1.5-flash` (Lower quota usage, good performance)

## 📝 Prompt Engineering

### Risk Analysis Prompts
```markdown
**Template**: Risk Exposure Analysis
**Purpose**: Generate concise portfolio risk briefings
**Format**: Voice-optimized, under 50 words
**Structure**: 
1. Current allocation vs previous
2. Key earnings surprises
3. Market sentiment assessment
4. Actionable recommendation
```

**Example Input**:
```json
{
  "exposure": {"current": 22, "previous": 18},
  "earnings_data": {"TSMC": {"surprise": 4.3}, "NVDA": {"surprise": 5.1}},
  "market_sentiment": "cautious"
}
```

**Example Output**:
> "Your Asia tech allocation is 22% of AUM, up from 18% yesterday. TSMC beat estimates by 4.3%, NVDA by 5.1%. Regional sentiment is cautious with rising yields pressuring valuations."

### Stock Analysis Prompts
```markdown
**Template**: Individual Stock Analysis
**Purpose**: Provide specific stock insights and recommendations
**Context**: Current price, earnings, market conditions
**Output**: Professional analysis with specific metrics
```

### General Financial Q&A
```markdown
**Template**: Financial Advisory
**Purpose**: Answer general financial questions
**Tone**: Professional but accessible
**Scope**: Market insights, portfolio strategy, risk management
```

## 🎯 AI Tool Applications

### 1. Risk Exposure Analysis
- **Input**: Portfolio data, market conditions
- **Processing**: Multi-factor risk assessment
- **Output**: Concise verbal briefing with actionable insights

### 2. Earnings Impact Analysis
- **Input**: Earnings surprises, estimate comparisons
- **Processing**: Contextual analysis with market implications
- **Output**: Investment impact assessment

### 3. Market Sentiment Processing
- **Input**: Market data, news sentiment, technical indicators
- **Processing**: Multi-source sentiment synthesis
- **Output**: Strategic positioning recommendations

### 4. Stock-Specific Insights
- **Input**: Individual stock data, user queries
- **Processing**: Fundamental and technical analysis
- **Output**: Targeted investment recommendations

## 🔄 Fallback Mechanisms

### Intelligent Fallbacks
When LLM APIs are unavailable, the system provides:

1. **Stock-Specific Responses**
   - NVDA: AI chip market leader context
   - TSMC: Semiconductor manufacturing insights
   - AAPL: Consumer tech positioning

2. **Risk Analysis Fallbacks**
   - Sample portfolio allocations
   - Mock earnings data
   - Standard risk assessments

3. **General Financial Guidance**
   - Principle-based advice
   - Educational content
   - Strategy frameworks

## 📊 Response Optimization

### Voice-Friendly Formatting
- **Length**: 30-60 words for voice delivery
- **Structure**: Clear beginning, middle, end
- **Language**: Conversational but professional
- **Numbers**: Formatted for speech synthesis

### Quality Assurance
- **Accuracy**: Cross-referenced with financial data
- **Relevance**: Contextual to user's portfolio/query
- **Timeliness**: Incorporates latest market data
- **Actionability**: Provides specific next steps

## 🚀 Performance Optimization

### Response Speed
- **Groq**: ~500ms average response time
- **Caching**: 5-minute cache for repeated queries
- **Async Processing**: Non-blocking API calls

### Rate Limiting
- **Groq**: 30 requests/minute (free tier)
- **Gemini**: 15 requests/minute (free tier)
- **Fallback**: No limits, instant response

## 🔐 Security & Privacy

### API Key Management
- Environment variables only
- No hardcoded credentials
- Secure cloud deployment practices

### Data Handling
- No persistent storage of user queries
- Ephemeral processing only
- Privacy-compliant logging

## 📈 AI Model Capabilities

### Groq (Llama 3.1 70B)
**Strengths**:
- Fast inference (sub-second)
- Strong reasoning capabilities
- Good financial domain knowledge
- Consistent formatting

**Use Cases**:
- Real-time portfolio analysis
- Complex financial reasoning
- Multi-factor risk assessment

### Gemini Flash
**Strengths**:
- Reliable and stable
- Good general knowledge
- Lower API costs
- Consistent availability

**Use Cases**:
- Backup processing
- General financial Q&A
- Market commentary

## 🛠️ Development Guidelines

### Adding New AI Features
1. **Define Use Case**: Specific financial analysis need
2. **Design Prompt**: Structured template with examples
3. **Test Responses**: Validate quality and consistency
4. **Add Fallbacks**: Ensure graceful degradation
5. **Monitor Performance**: Track success rates and quality

### Prompt Best Practices
- **Specificity**: Clear role and context
- **Examples**: Include desired output format
- **Constraints**: Length, tone, structure requirements
- **Flexibility**: Handle various input scenarios

### Error Handling
- **Graceful Degradation**: Never return empty responses
- **Context Preservation**: Maintain conversation flow
- **User Communication**: Clear status updates
- **Fallback Quality**: Useful alternative responses

## 📋 Usage Examples

### Risk Analysis Query
```bash
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our Asia tech risk exposure today?"}'
```

### Stock Analysis Query
```bash
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze NVDA stock performance and outlook"}'
```

### Earnings Query
```bash
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Any significant earnings surprises this week?"}'
```

## 🔍 Monitoring & Analytics

### Key Metrics
- **Response Time**: Average API response latency
- **Success Rate**: Percentage of successful AI calls
- **Fallback Usage**: How often fallbacks are triggered
- **User Satisfaction**: Quality of generated responses

### Logging
```python
# Example log structure
{
  "timestamp": "2025-05-29T10:30:00Z",
  "query": "NVDA risk analysis",
  "provider": "groq",
  "response_time": 487,
  "status": "success",
  "tokens_used": 245
}
```

## 🔄 Continuous Improvement

### Model Updates
- Monitor new model releases
- Test performance improvements
- Gradual rollout of updates
- Backward compatibility maintenance

### Prompt Optimization
- A/B testing of prompt variations
- User feedback integration
- Performance metric analysis
- Iterative refinement

---

## 📞 Support & Troubleshooting

### Common Issues
1. **"None" Response**: Check API keys and service status
2. **Slow Responses**: Verify network connectivity
3. **Poor Quality**: Review prompt templates
4. **Rate Limits**: Implement backoff strategies

### Debug Endpoints
- `GET /debug` - Check provider status
- `GET /health` - Service health check
- `POST /test-chat` - Direct provider testing

### Getting Help
- Check service logs for detailed error messages
- Verify environment variable configuration
- Test individual components separately
- Review API provider status pages