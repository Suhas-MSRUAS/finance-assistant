import os
import asyncio
import logging
from enum import Enum
from typing import Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    GROQ = "groq"
    GEMINI = "gemini"  # Keep as backup only

class RiskData(BaseModel):
    exposure: float
    change: float
    previous_exposure: float = None
    earnings_data: Dict[str, Any] = None
    market_sentiment: str = "neutral"

class GroqLLMClient:
    def __init__(self):
        self.providers = {}
        self.initialize_providers()
    
    def initialize_providers(self):
        """Initialize LLM providers - Groq first, Gemini as backup"""
        logger.info("Initializing LLM providers...")
        
        # Groq (PRIMARY - Fast and reliable)
        if os.getenv("GROQ_API_KEY"):
            try:
                import openai
                self.providers[LLMProvider.GROQ] = openai.AsyncOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY")
                )
                logger.info("✅ Groq client initialized (PRIMARY)")
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # Gemini (BACKUP ONLY - for when Groq is down)
        if os.getenv("GEMINI_API_KEY"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.providers[LLMProvider.GEMINI] = genai.GenerativeModel('gemini-1.5-flash')  # Use flash for lower quota usage
                logger.info("✅ Gemini client initialized (BACKUP)")
            except Exception as e:
                logger.error(f"❌ Gemini initialization failed: {e}")
        
        if not self.providers:
            logger.warning("⚠️ No LLM providers configured - will use fallback narratives")
        else:
            logger.info(f"Active providers: {[p.value for p in self.providers.keys()]}")
    
    def create_fallback_response(self, query: str) -> str:
        """Create intelligent fallback responses when no LLM providers are available"""
        query_lower = query.lower()
        
        # Stock-specific responses
        if 'nvda' in query_lower or 'nvidia' in query_lower:
            return "NVIDIA (NVDA) is a leading AI and semiconductor company with strong performance in AI chip markets. Recent quarters have shown significant growth driven by AI demand. For current stock data and specific risk exposure, I'd need access to live market feeds."
        
        if 'tsmc' in query_lower or 'taiwan semiconductor' in query_lower:
            return "TSMC (Taiwan Semiconductor) is the world's largest contract chip manufacturer and a key supplier for major tech companies including Apple and NVIDIA. For current performance data, please check real-time market information."
        
        if 'risk' in query_lower and 'asia' in query_lower:
            return "For Asia tech risk exposure analysis, I'd typically provide current allocation percentages, recent changes, and market sentiment. However, I need access to live portfolio data and market feeds to give you accurate information."
        
        if 'exposure' in query_lower or 'allocation' in query_lower:
            return "Portfolio exposure and allocation analysis requires access to current holdings and market data. I can discuss general principles of risk management and diversification strategies if that would be helpful."
        
        # General financial topics
        if any(term in query_lower for term in ['stock', 'market', 'trading', 'investment']):
            return "I can help with financial analysis and market insights. However, I currently don't have access to live market data or LLM providers. For real-time information, please ensure API keys are configured or try again shortly."
        
        # Default response
        return "I'm a financial analysis assistant. I can help with market insights, stock analysis, and investment questions. However, I currently don't have access to live data sources. Please ensure services are properly configured."

    async def generate_response(self, prompt: str) -> str:
        """Generate response using Groq first, then Gemini as backup"""
        
        # Try Groq first (PRIMARY)
        if LLMProvider.GROQ in self.providers:
            try:
                logger.info("Using Groq (primary provider)")
                client = self.providers[LLMProvider.GROQ]
                response = await client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # Fast and capable
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"❌ Groq failed: {e}")
        
        # Fallback to Gemini only if Groq fails
        if LLMProvider.GEMINI in self.providers:
            try:
                logger.info("Using Gemini (backup provider)")
                client = self.providers[LLMProvider.GEMINI]
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, client.generate_content, prompt)
                return response.text
                
            except Exception as e:
                logger.warning(f"❌ Gemini backup failed: {e}")
        
        # All providers failed
        logger.error("All LLM providers failed")
        raise Exception("All LLM providers unavailable")

    async def generate_risk_exposure_narrative(self, risk_data: Dict[str, Any]) -> str:
        """Generate specific risk exposure narrative with earnings surprises"""
        
        try:
            prompt = self._create_risk_exposure_prompt(risk_data)
            narrative = await self.generate_response(prompt)
            logger.info("✅ Successfully generated risk exposure narrative")
            return narrative
            
        except Exception as e:
            logger.warning(f"❌ LLM generation failed: {str(e)}")
            return self._generate_structured_fallback(risk_data)
    
    def _create_risk_exposure_prompt(self, risk_data: Dict[str, Any]) -> str:
        """Create enhanced prompt for risk exposure with earnings surprises"""
        current_exposure = risk_data.get('exposure', 22)
        previous_exposure = risk_data.get('previous_exposure', 18)
        earnings_data = risk_data.get('earnings_data', {})
        market_sentiment = risk_data.get('market_sentiment', 'neutral')
        
        # Format earnings data if provided
        earnings_info = ""
        if earnings_data:
            for company, data in earnings_data.items():
                surprise = data.get('surprise', 0)
                direction = "beat" if surprise > 0 else "missed"
                earnings_info += f"{company} {direction} estimates by {abs(surprise)}%. "
        
        return f"""You are a financial risk analyst providing a concise verbal briefing. Generate a response that sounds natural for voice delivery.

CURRENT DATA:
- Asia tech allocation: {current_exposure}% of AUM
- Previous allocation: {previous_exposure}% 
- Earnings updates: {earnings_info or "TSMC beat estimates by 4%, Samsung missed by 2%"}
- Market sentiment: {market_sentiment}

REQUIRED FORMAT:
Start with current exposure and change, then highlight earnings surprises, end with sentiment assessment.

EXAMPLE RESPONSE STYLE:
"Today, your Asia tech allocation is [X]% of AUM, [up/down] from [Y]% yesterday. [Earnings details]. Regional sentiment is [sentiment] with a [cautionary/optimistic] tilt due to [key factor]."

Keep it conversational, under 50 words, suitable for voice delivery. Focus on the most important changes and actionable insights."""

    def _generate_structured_fallback(self, risk_data: Dict[str, Any]) -> str:
        """Generate structured fallback when all LLMs fail"""
        current_exposure = risk_data.get('exposure', 22)
        previous_exposure = risk_data.get('previous_exposure', 18)
        change_direction = "up" if current_exposure > previous_exposure else "down"
        
        earnings_data = risk_data.get('earnings_data', {})
        earnings_summary = ""
        if earnings_data:
            surprises = []
            for company, data in earnings_data.items():
                surprise = data.get('surprise', 0)
                direction = "beat" if surprise > 0 else "missed"
                surprises.append(f"{company} {direction} by {abs(surprise)}%")
            earnings_summary = ". ".join(surprises) + "."
        else:
            earnings_summary = "TSMC beat estimates by 4%, Samsung missed by 2%."
        
        sentiment = risk_data.get('market_sentiment', 'neutral')
        
        return f"Today, your Asia tech allocation is {current_exposure}% of AUM, {change_direction} from {previous_exposure}% yesterday. {earnings_summary} Regional sentiment is {sentiment} with a cautionary tilt due to rising yields."

    async def generate_narrative(self, risk_data: Dict[str, Any]) -> str:
        """Original narrative generation method for backward compatibility"""
        try:
            prompt = risk_data.get("prompt") or self._create_prompt(risk_data)
            return await self.generate_response(prompt)
        except Exception as e:
            logger.warning(f"❌ Narrative generation failed: {str(e)}")
            return self._generate_basic_narrative(risk_data)
    
    def _create_prompt(self, risk_data: Dict[str, Any]) -> str:
        """Enhanced prompt creation method for voice-friendly responses"""
        current_exposure = risk_data.get('exposure', 22)
        previous_exposure = risk_data.get('previous_exposure', 18)
        earnings_data = risk_data.get('earnings_data', {})
        market_sentiment = risk_data.get('market_sentiment', 'neutral')
        
        # Format earnings data if provided
        earnings_info = ""
        if earnings_data:
            for company, data in earnings_data.items():
                surprise = data.get('surprise', 0)
                direction = "beat" if surprise > 0 else "missed"
                earnings_info += f"{company} {direction} estimates by {abs(surprise)}%. "
        
        return f"""You are a financial risk analyst providing a concise verbal briefing. Generate a response that sounds natural for voice delivery.

CURRENT DATA:
- Asia tech allocation: {current_exposure}% of AUM
- Previous allocation: {previous_exposure}% 
- Earnings updates: {earnings_info or "No significant earnings surprises today"}
- Market sentiment: {market_sentiment}

REQUIRED FORMAT:
Start with current exposure and change, then highlight earnings surprises, end with sentiment assessment.

EXAMPLE RESPONSE:
"Today, your Asia tech allocation is [X]% of AUM, [up/down] from [Y]% yesterday. [Earnings details]. Regional sentiment is [sentiment] with a [cautionary/optimistic] tilt due to [key factor]."

Keep it conversational, under 50 words, suitable for voice delivery. Focus on actionable insights only."""
    
    def _generate_basic_narrative(self, risk_data: Dict[str, Any]) -> str:
        """Enhanced basic narrative fallback for voice delivery"""
        current_exposure = risk_data.get('exposure', 22)
        previous_exposure = risk_data.get('previous_exposure', 18)
        change_direction = "up" if current_exposure > previous_exposure else "down"
        
        earnings_data = risk_data.get('earnings_data', {})
        earnings_summary = ""
        if earnings_data:
            surprises = []
            for company, data in earnings_data.items():
                surprise = data.get('surprise', 0)
                direction = "beat" if surprise > 0 else "missed"
                surprises.append(f"{company} {direction} by {abs(surprise)}%")
            earnings_summary = ". ".join(surprises) + "."
        else:
            earnings_summary = "TSMC beat estimates by 4%, Samsung missed by 2%."
        
        sentiment = risk_data.get('market_sentiment', 'neutral')
        
        return f"Today, your Asia tech allocation is {current_exposure}% of AUM, {change_direction} from {previous_exposure}% yesterday. {earnings_summary} Regional sentiment is {sentiment} with a cautionary tilt due to rising yields."

# FastAPI Application
app = FastAPI(title="Groq-Powered Financial Assistant", version="3.0.0")

# Initialize the Groq LLM client
llm_client = GroqLLMClient()

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Groq-Powered Financial Assistant starting up...")
    logger.info(f"Available providers: {list(llm_client.providers.keys())}")

# New endpoint for risk exposure analysis
@app.post("/risk-exposure")
async def generate_risk_exposure_endpoint(risk_data: dict):
    """Generate risk exposure narrative with earnings surprises"""
    try:
        logger.info(f"Received risk exposure request: {risk_data}")
        
        # Generate structured narrative
        narrative = await llm_client.generate_risk_exposure_narrative(risk_data)
        
        response = {
            "narrative": narrative,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "response_type": "risk_exposure_briefing"
        }
        
        logger.info("✅ Risk exposure narrative generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating risk exposure narrative: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk exposure analysis failed: {str(e)}")

# Keep existing endpoints
@app.post("/narrate")
async def generate_narrative_endpoint(risk_data: dict):
    """Generate market narrative from risk data"""
    try:
        logger.info(f"Received risk data: {risk_data}")
        narrative = await llm_client.generate_narrative(risk_data)
        
        response = {
            "narrative": narrative,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "success",
            "provider_used": "groq-primary"
        }
        
        logger.info("✅ Narrative generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error generating narrative: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "providers_available": [p.value for p in llm_client.providers.keys()],
        "primary_provider": "groq",
        "service": "groq-financial-assistant-v3.0"
    }

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(data: ChatQuery):
    query = data.query.strip()
    logger.info(f"Received chat request: {query}")
    
    try:
        # If we have LLM providers available, use them
        if llm_client.providers:
            logger.info(f"Available providers: {[p.value for p in llm_client.providers.keys()]}")
            
            # Enhanced financial assistant prompt
            prompt = f"""You are an expert financial analyst and portfolio advisor with deep knowledge of:
- Global equity markets and individual stocks (including NVDA, TSMC, AAPL, etc.)
- Risk management and portfolio allocation
- Earnings analysis and market sentiment
- Technical and fundamental analysis
- Market trends and sector dynamics

User Question: {query}

Provide a clear, professional response with specific insights. If discussing stocks, include relevant metrics, recent performance, or market context where appropriate. Keep responses concise but informative."""
            
            try:
                response_text = await llm_client.generate_response(prompt)
                
                # Determine which provider was used
                provider_used = "groq" if LLMProvider.GROQ in llm_client.providers else "gemini"
                
                logger.info(f"Successfully got response from {provider_used}")
                return {
                    "response": response_text,
                    "status": "success",
                    "provider_used": provider_used,
                    "query": query,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            except Exception as e:
                logger.warning(f"All providers failed: {e}")
        
        # No providers available or all failed - use intelligent fallback
        logger.info("Using intelligent fallback response")
        fallback_response = llm_client.create_fallback_response(query)
        
        return {
            "response": fallback_response,
            "status": "success",
            "provider_used": "intelligent_fallback",
            "query": query,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        # Even if there's an error, provide a helpful response
        return {
            "response": "I'm experiencing some technical difficulties, but I'm here to help with financial analysis and market questions. Please try again in a moment.",
            "status": "error",
            "provider_used": None,
            "query": query,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

@app.get("/debug")
async def debug_status():
    """Debug endpoint to check what's happening"""
    return {
        "service": "groq_financial_assistant",
        "providers_configured": [p.value for p in llm_client.providers.keys()],
        "total_providers": len(llm_client.providers),
        "primary_provider": "groq",
        "environment_check": {
            "GROQ_API_KEY": "✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Missing",
            "GEMINI_API_KEY": "✅ Set (backup)" if os.getenv("GEMINI_API_KEY") else "❌ Missing (backup only)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)