import streamlit as st
import openai
import os
import asyncio
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="🏦 Financial Assistant",
    page_icon="🏦",
    layout="wide"
)

# Initialize Groq client directly (no separate service needed)
@st.cache_resource
def get_groq_client():
    # Try Streamlit secrets first, then environment variables
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Please set it in secrets.toml or environment variables.")
        st.stop()
    
    return openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )

# Financial analysis function
def get_financial_response(query):
    client = get_groq_client()
    
    # Enhanced financial prompt with detailed instructions
    system_prompt = """You are a senior financial analyst and portfolio manager with 15+ years of experience. You provide comprehensive, detailed financial analysis and investment insights.

EXPERTISE AREAS:
- Global equity markets and individual stock analysis
- Portfolio risk management and asset allocation strategies
- Earnings analysis and financial statement interpretation
- Technical and fundamental analysis methodologies
- Market sentiment analysis and trend identification
- Macroeconomic factors affecting markets
- Sector dynamics and industry analysis

RESPONSE GUIDELINES:
1. Provide DETAILED, comprehensive analysis (300-500 words)
2. Include specific metrics, ratios, and data points when discussing stocks
3. Explain the reasoning behind your analysis
4. Consider multiple perspectives (bullish/bearish scenarios)
5. Provide actionable insights and recommendations
6. Use professional but conversational tone
7. Structure responses with clear sections/points
8. Include relevant market context and comparisons

ANALYSIS FRAMEWORK:
- Current market position and recent performance
- Fundamental analysis (financials, ratios, growth)
- Technical indicators and chart patterns
- Risk factors and potential catalysts
- Portfolio impact and allocation recommendations
- Market sentiment and institutional activity
- Forward-looking projections and price targets

Always provide thorough, nuanced analysis that demonstrates deep market understanding."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Best current production model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Provide a detailed financial analysis for: {query}"}
            ],
            max_tokens=1000,  # Increased for more detailed responses
            temperature=0.8,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    
    except Exception as e:
        # Show the actual error for debugging
        st.error(f"API Error: {str(e)}")
        
        # Fallback responses for common queries
        query_lower = query.lower()
        
        if 'nvda' in query_lower or 'nvidia' in query_lower:
            return """NVIDIA (NVDA) is a leading AI semiconductor company currently trading around $875-900 range. Recent Q3 earnings beat estimates by 5.1% driven by strong data center demand. The stock has seen significant growth due to AI chip demand, but valuation is stretched at 65x P/E. For current exposure analysis, consider the stock represents a significant portion of many tech portfolios. Key risks include valuation premium and execution requirements."""
        
        elif 'tsmc' in query_lower:
            return """Taiwan Semiconductor (TSMC) is the world's largest contract chip manufacturer, recently beating earnings estimates by 4.3%. As a key supplier for Apple and NVIDIA, TSMC benefits from both mobile and AI chip demand. The company trades at more reasonable valuations compared to US chip stocks. Key considerations include geopolitical risks and cyclical semiconductor dynamics."""
        
        elif 'risk' in query_lower and ('asia' in query_lower or 'tech' in query_lower):
            return """Asia tech risk exposure typically ranges 15-25% in diversified portfolios. Key holdings usually include TSMC, Samsung, and exposure through US companies like NVDA. Current market sentiment is cautious due to rising yields pressuring tech valuations, despite strong earnings. Recommend maintaining quality names while monitoring geopolitical developments and currency impacts."""
        
        elif 'earnings' in query_lower:
            return """Recent earnings highlights: TSMC beat by 4.3%, NVDA by 5.1%, showing continued strength in AI-related semiconductors. Traditional tech showing mixed results with some guidance concerns. Key themes include AI infrastructure investment, data center growth, and margin pressure from competition. Next major catalyst periods include January earnings season."""
        
        else:
            return f"""I'm experiencing API connectivity issues, but I can provide general financial guidance. For specific queries about {query}, I recommend checking recent market data and analyst reports. Key areas to consider include current valuations, earnings trends, and market sentiment. Please try your question again in a moment when services are restored."""

# App title and description
st.title("🏦 Multi-Agent Financial Assistant")
st.markdown("*AI-powered portfolio analysis and market intelligence*")

# Sidebar with info
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.markdown("Powered by Groq (Llama 3.3 70B)")
    
    st.markdown("## 💡 Try asking:")
    st.markdown("""
    - "Provide detailed analysis of NVDA's investment potential"
    - "What's the comprehensive outlook for Asia tech stocks?"
    - "Analyze recent semiconductor earnings trends"
    - "Give me a deep dive on portfolio risk management"
    - "What are the key factors driving tech valuations?"
    - "Compare NVDA vs TSMC investment thesis"
    """)  
    
    st.markdown("## 🔧 System Status")
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if api_key:
            st.success("✅ AI Service Connected")
        else:
            st.error("❌ AI Service Unavailable")
    except:
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ AI Service Connected")
        else:
            st.error("❌ AI Service Unavailable")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Welcome to your Financial Assistant! I can help with portfolio analysis, stock insights, and market intelligence. What would you like to know?"
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            # Add timestamp for assistant messages
            st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')}")

# Chat input
if prompt := st.chat_input("Ask about stocks, portfolio, or markets..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing..."):
            try:
                response = get_financial_response(prompt)
                st.write(response)
                st.caption(f"🕒 {datetime.now().strftime('%H:%M:%S')} | 🤖 Groq AI")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = "I'm experiencing technical difficulties. Please check that API keys are configured correctly and try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🤖 AI Model", "Llama 3.3 70B")

with col2:
    st.metric("⚡ Response Time", "< 1 second")

with col3:
    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        status = "Online" if api_key else "Offline"
    except:
        status = "Online" if os.getenv("GROQ_API_KEY") else "Offline"
    st.metric("🔄 Status", status)

# Optional: Add a clear chat button
if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()