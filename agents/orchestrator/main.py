from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import aiohttp
from typing import Optional
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Market Brief Service", version="2.0.0")

class MarketData(BaseModel):
    today: float
    yesterday: float
    additional_context: Optional[str] = None

class BriefResponse(BaseModel):
    narrative: str
    risk_exposure: float
    change: float
    timestamp: str

class ChatQuery(BaseModel):
    query: str

@app.get("/brief", response_model=BriefResponse)
async def market_brief():
    """Generate market brief with detailed error logging"""
    try:
        logger.info("Starting market brief generation...")

        async with aiohttp.ClientSession() as session:
            # Get risk exposure
            logger.info("Calling risk exposure service...")
            try:
                risk_data = await get_risk_exposure(session, 22, 18)
                logger.info(f"Risk data received: {risk_data}")
            except Exception as e:
                logger.error(f"Risk exposure failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Risk service error: {str(e)}")

            # Generate narrative
            logger.info("Calling narrative service...")
            try:
                narrative_data = await generate_narrative(session, risk_data)
                logger.info(f"Narrative data received: {narrative_data}")
            except Exception as e:
                logger.error(f"Narrative service failed: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Narrative service error: {str(e)}")

            return BriefResponse(
                narrative=narrative_data["narrative"],
                risk_exposure=risk_data["exposure"],
                change=risk_data["change"],
                timestamp=narrative_data.get("timestamp", "")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in market_brief: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/brief/custom", response_model=BriefResponse)
async def custom_market_brief(data: MarketData):
    """Generate market brief with custom data"""
    try:
        async with aiohttp.ClientSession() as session:
            risk_data = await get_risk_exposure(session, data.today, data.yesterday)

            # Enhanced prompt with additional context
            prompt = create_enhanced_prompt(risk_data, data.additional_context)
            narrative_data = await generate_narrative(session, risk_data, prompt)

            return BriefResponse(
                narrative=narrative_data["narrative"],
                risk_exposure=risk_data["exposure"],
                change=risk_data["change"],
                timestamp=narrative_data.get("timestamp", "")
            )

    except Exception as e:
        logger.error(f"Error in custom brief: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def get_risk_exposure(session: aiohttp.ClientSession, today: float, yesterday: float) -> dict:
    """Get risk exposure data from risk service"""
    async with session.post(
        "http://localhost:8003/risk_exposure",
        json={"today": today, "yesterday": yesterday, "asset_class": "Asia Tech"},
        timeout=aiohttp.ClientTimeout(total=5)
    ) as response:
        response.raise_for_status()
        data = await response.json()
        logger.info(f"Received from risk service: {data}")
        
        # Extract the correct fields from the nested response structure
        if 'exposure' in data and isinstance(data['exposure'], dict):
            exposure_data = {
                'exposure': data['exposure']['current'],
                'change': data['exposure']['change_percentage']
            }
            logger.info(f"Extracted exposure data: {exposure_data}")
            return exposure_data
        else:
            raise ValueError("Invalid response structure from risk service")

async def generate_narrative(session: aiohttp.ClientSession, risk_data: dict, custom_prompt: str = None) -> dict:
    """Generate narrative from narrative service"""
    prompt = custom_prompt or create_default_prompt(risk_data)

    async with session.post(
        "http://localhost:8004/narrate",
        json={"prompt": prompt},
        timeout=aiohttp.ClientTimeout(total=10)
    ) as response:
        response.raise_for_status()
        return await response.json()

def create_default_prompt(risk_data: dict) -> str:
    """Create default market brief prompt"""
    exposure = risk_data.get('exposure', 'unknown')
    change = risk_data.get('change', 'unknown')
    return (
        f"Asia tech exposure is {exposure}%, "
        f"change is {change}%. "
        f"TSMC beat estimates by 4%. Samsung missed by 2%. "
        f"Provide market implications and outlook."
    )

def create_enhanced_prompt(risk_data: dict, additional_context: str = None) -> str:
    """Create enhanced prompt with additional context"""
    base_prompt = create_default_prompt(risk_data)
    if additional_context:
        return f"{base_prompt} Additional context: {additional_context}"
    return base_prompt

async def get_stock_data(session: aiohttp.ClientSession, symbol: str) -> dict:
    """Get stock data from API agent"""
    try:
        async with session.get(
            f"http://localhost:8000/stock/{symbol}/quote",
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        raise

async def call_language_agent(session: aiohttp.ClientSession, query: str) -> dict:
    """Call the language agent for general financial questions"""
    try:
        async with session.post(
            "http://localhost:8004/chat",
            json={"query": query},
            timeout=aiohttp.ClientTimeout(total=15)
        ) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error calling language agent: {e}")
        raise

def is_stock_question(query: str) -> Optional[str]:
    """Check if query is about a specific stock and extract symbol"""
    query_lower = query.lower()
    
    # Skip if it's clearly about portfolio/risk exposure rather than individual stocks
    portfolio_keywords = ['portfolio', 'exposure', 'allocation', 'our risk', 'our position']
    if any(keyword in query_lower for keyword in portfolio_keywords):
        return None
    
    # Common stock symbols and their variations
    stock_mappings = {
        'nvda': 'NVDA', 'nvidia': 'NVDA',
        'tsmc': 'TSM', 'taiwan semiconductor': 'TSM',
        'aapl': 'AAPL', 'apple': 'AAPL',
        'msft': 'MSFT', 'microsoft': 'MSFT',
        'googl': 'GOOGL', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amzn': 'AMZN', 'amazon': 'AMZN',
        'meta': 'META', 'facebook': 'META',
        'tesla': 'TSLA', 'tsla': 'TSLA',
        'samsung': '005930.KS'
    }
    
    # Only check mappings if the query seems to be asking about a specific company
    company_indicators = ['stock', 'share', 'price of', 'how is', 'what about', 'performance of']
    if any(indicator in query_lower for indicator in company_indicators):
        for key, symbol in stock_mappings.items():
            if key in query_lower:
                return symbol
    
    # Check for stock symbol patterns only if it looks like a stock query
    if any(word in query_lower for word in ['stock', 'ticker', 'symbol', 'quote']):
        import re
        # More restrictive pattern - avoid common words
        symbol_pattern = r'\b([A-Z]{3,5})\b'
        matches = re.findall(symbol_pattern, query.upper())
        
        # Filter out common words that aren't stock symbols
        excluded_words = {'WHAT', 'OUR', 'THE', 'AND', 'FOR', 'ARE', 'YOU', 'CAN', 'HOW', 'WHY', 'WHO', 'WHEN', 'WHERE'}
        valid_matches = [match for match in matches if match not in excluded_words]
        
        if valid_matches:
            return valid_matches[0]
    
    return None

@app.post("/chat")
async def chat_endpoint(data: ChatQuery):
    """Enhanced chat endpoint supporting all financial questions"""
    user_question = data.query.strip()
    logger.info(f"Received chat query: {user_question}")
    
    try:
        async with aiohttp.ClientSession() as session:
            
            # Check if it's a specific stock question
            stock_symbol = is_stock_question(user_question)
            if stock_symbol:
                logger.info(f"Detected stock question for symbol: {stock_symbol}")
                try:
                    # Get stock data
                    stock_data = await get_stock_data(session, stock_symbol)
                    
                    # Create enhanced prompt with stock data
                    enhanced_query = f"""
                    User asked: {user_question}
                    
                    Current stock data for {stock_symbol}:
                    - Price: ${stock_data.get('price', 'N/A')}
                    - Change: {stock_data.get('change', 'N/A')} ({stock_data.get('change_percent', 'N/A')})
                    - Volume: {stock_data.get('volume', 'N/A'):,}
                    - High: ${stock_data.get('high', 'N/A')}
                    - Low: ${stock_data.get('low', 'N/A')}
                    
                    Please provide a comprehensive analysis addressing the user's question with this current data.
                    """
                    
                    # Call language agent with enhanced prompt
                    response = await call_language_agent(session, enhanced_query)
                    return {"response": response.get("response", "Unable to analyze stock data")}
                    
                except Exception as e:
                    logger.warning(f"Stock data fetch failed for {stock_symbol}: {e}")
                    # Fall back to general language agent
                    response = await call_language_agent(session, user_question)
                    return {"response": response.get("response", "Unable to process stock question")}
            
            # Check if it's about Asia tech risk specifically
            elif "risk" in user_question.lower() and "asia" in user_question.lower() and "tech" in user_question.lower():
                logger.info("Processing Asia tech risk question")
                try:
                    # Call risk exposure service with live data
                    risk_data = await get_risk_exposure(session, 22, 18)
                    
                    # Call narrative service with the risk data
                    narrative_data = await generate_narrative(session, risk_data)
                    
                    return {"response": narrative_data.get("narrative", "Unable to generate risk analysis")}
                    
                except Exception as e:
                    logger.error(f"Risk analysis failed: {e}")
                    return {"response": "Sorry, I couldn't retrieve the latest risk analysis right now."}
            
            # For all other financial questions, use the language agent
            else:
                logger.info("Processing general financial question")
                try:
                    response = await call_language_agent(session, user_question)
                    return {"response": response.get("response", "Unable to process your question")}
                    
                except Exception as e:
                    logger.error(f"Language agent failed: {e}")
                    return {"response": "Sorry, I'm having trouble processing your question right now. Please try again."}
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        return {"response": "I encountered an unexpected error. Please try again."}

@app.post("/agent")
async def handle_agent_query(data: ChatQuery):
    """Flexible orchestration entry point for external agents"""
    user_question = data.query.strip()
    logger.info(f"Received agent query: {user_question}")

    # Route to appropriate chat handling
    return await chat_endpoint(data)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with aiohttp.ClientSession() as session:
            # Check all dependent services
            services = {
                "api_agent": "http://localhost:8000/health",
                "analysis_agent": "http://localhost:8003/health", 
                "language_agent": "http://localhost:8004/health"
            }
            
            service_status = {}
            for service_name, url in services.items():
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        service_status[service_name] = "healthy" if resp.status == 200 else "unhealthy"
                except:
                    service_status[service_name] = "unreachable"
            
            overall_healthy = all(status == "healthy" for status in service_status.values())
            
            return {
                "status": "healthy" if overall_healthy else "degraded",
                "services": service_status,
                "version": "2.0.0"
            }
    except Exception as e:
        logger.error("Health check failed", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {}
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)