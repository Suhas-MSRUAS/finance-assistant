from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import requests
import os
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Stock Data API",
    description="Comprehensive stock market data API with Alpha Vantage integration",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
CACHE_TTL = 300  # 5 minutes cache
RATE_LIMIT_DELAY = 12  # 12 seconds between API calls (Alpha Vantage free tier: 5 calls/minute)

# Enums for API functions
class TimeSeriesFunction(str, Enum):
    DAILY = "TIME_SERIES_DAILY"
    WEEKLY = "TIME_SERIES_WEEKLY"
    MONTHLY = "TIME_SERIES_MONTHLY"
    INTRADAY = "TIME_SERIES_INTRADAY"

class OutputSize(str, Enum):
    COMPACT = "compact"  # 100 data points
    FULL = "full"       # Full historical data

class Interval(str, Enum):
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    MIN_60 = "60min"

# Pydantic models
class StockQuote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: str
    volume: int
    timestamp: str

class TimeSeriesData(BaseModel):
    date: str
    open: float = Field(..., alias="1. open")
    high: float = Field(..., alias="2. high") 
    low: float = Field(..., alias="3. low")
    close: float = Field(..., alias="4. close")
    volume: int = Field(..., alias="5. volume")

class CompanyOverview(BaseModel):
    symbol: str
    name: str
    exchange: str
    currency: str
    country: str
    sector: str
    industry: str
    market_cap: Optional[str] = None
    pe_ratio: Optional[str] = None
    dividend_yield: Optional[str] = None

# Cache for API responses
response_cache = {}
last_request_time = {}

class AlphaVantageClient:
    """Enhanced Alpha Vantage API client with error handling and rate limiting"""
    
    def __init__(self):
        if not API_KEY:
            raise ValueError("ALPHAVANTAGE_API_KEY environment variable not set")
        self.api_key = API_KEY
        self.base_url = BASE_URL
    
    def _make_request(self, params: Dict) -> Dict:
        """Make rate-limited API request with error handling"""
        
        # Rate limiting
        current_time = time.time()
        if hasattr(self, '_last_request_time'):
            time_since_last = current_time - self._last_request_time
            if time_since_last < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - time_since_last)
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                raise HTTPException(status_code=400, detail=data["Error Message"])
            
            if "Note" in data:
                raise HTTPException(status_code=429, detail="API call frequency limit reached")
            
            if "Information" in data and "Thank you for using Alpha Vantage" in data["Information"]:
                raise HTTPException(status_code=429, detail="API rate limit exceeded")
            
            self._last_request_time = current_time
            return data
            
        except requests.exceptions.Timeout:
            raise HTTPException(status_code=408, detail="Request timeout")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"API service unavailable: {str(e)}")
    
    @lru_cache(maxsize=100, typed=True)
    def get_time_series(self, symbol: str, function: str, outputsize: str = "compact", 
                       interval: Optional[str] = None) -> Dict:
        """Get time series data with caching"""
        params = {
            "function": function,
            "symbol": symbol.upper(),
            "outputsize": outputsize
        }
        
        if interval and function == "TIME_SERIES_INTRADAY":
            params["interval"] = interval
        
        return self._make_request(params)
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol.upper()
        }
        return self._make_request(params)
    
    def get_company_overview(self, symbol: str) -> Dict:
        """Get company overview"""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper()
        }
        return self._make_request(params)
    
    def search_symbol(self, keywords: str) -> Dict:
        """Search for symbols"""
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords
        }
        return self._make_request(params)

# Initialize client
try:
    alpha_vantage = AlphaVantageClient()
except ValueError as e:
    logger.error(f"Failed to initialize Alpha Vantage client: {e}")
    alpha_vantage = None

def get_client():
    """Dependency to get Alpha Vantage client"""
    if alpha_vantage is None:
        raise HTTPException(status_code=503, detail="API service not available - missing API key")
    return alpha_vantage

@app.get("/stock/{symbol}/quote", 
         summary="Get real-time stock quote",
         response_model=Dict)
def get_stock_quote(symbol: str, client: AlphaVantageClient = Depends(get_client)):
    """Get real-time stock quote for a symbol"""
    try:
        data = client.get_quote(symbol)
        
        if "Global Quote" not in data:
            raise HTTPException(status_code=404, detail=f"Quote data not found for symbol: {symbol}")
        
        quote_data = data["Global Quote"]
        
        # Parse and format the response
        formatted_quote = {
            "symbol": quote_data.get("01. symbol", symbol),
            "price": float(quote_data.get("05. price", 0)),
            "change": float(quote_data.get("09. change", 0)),
            "change_percent": quote_data.get("10. change percent", "0%"),
            "volume": int(quote_data.get("06. volume", 0)),
            "open": float(quote_data.get("02. open", 0)),
            "high": float(quote_data.get("03. high", 0)),
            "low": float(quote_data.get("04. low", 0)),
            "previous_close": float(quote_data.get("08. previous close", 0)),
            "latest_trading_day": quote_data.get("07. latest trading day", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        return formatted_quote
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stock/{symbol}/timeseries", 
         summary="Get time series data")
def get_time_series(
    symbol: str,
    function: TimeSeriesFunction = TimeSeriesFunction.DAILY,
    outputsize: OutputSize = OutputSize.COMPACT,
    interval: Optional[Interval] = None,
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Limit number of data points"),
    client: AlphaVantageClient = Depends(get_client)
):
    """Get time series data for a stock symbol"""
    try:
        data = client.get_time_series(
            symbol=symbol,
            function=function.value,
            outputsize=outputsize.value,
            interval=interval.value if interval else None
        )
        
        # Determine the time series key
        time_series_keys = [k for k in data.keys() if "Time Series" in k]
        if not time_series_keys:
            raise HTTPException(status_code=404, detail=f"Time series data not found for symbol: {symbol}")
        
        time_series_key = time_series_keys[0]
        time_series_data = data[time_series_key]
        
        # Process and format data
        formatted_data = []
        for date_str, values in time_series_data.items():
            try:
                formatted_entry = {
                    "date": date_str,
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "volume": int(values.get("5. volume", 0))
                }
                formatted_data.append(formatted_entry)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid data point for {date_str}: {e}")
                continue
        
        # Sort by date (most recent first)
        formatted_data.sort(key=lambda x: x["date"], reverse=True)
        
        # Apply limit if specified
        if limit:
            formatted_data = formatted_data[:limit]
        
        return {
            "symbol": symbol.upper(),
            "function": function.value,
            "outputsize": outputsize.value,
            "interval": interval.value if interval else None,
            "data_points": len(formatted_data),
            "data": formatted_data,
            "metadata": data.get("Meta Data", {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching time series for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stock/{symbol}/overview",
         summary="Get company overview")
def get_company_overview(symbol: str, client: AlphaVantageClient = Depends(get_client)):
    """Get comprehensive company overview"""
    try:
        data = client.get_company_overview(symbol)
        
        if not data or "Symbol" not in data:
            raise HTTPException(status_code=404, detail=f"Company overview not found for symbol: {symbol}")
        
        # Format overview data
        overview = {
            "symbol": data.get("Symbol", ""),
            "name": data.get("Name", ""),
            "description": data.get("Description", ""),
            "exchange": data.get("Exchange", ""),
            "currency": data.get("Currency", ""),
            "country": data.get("Country", ""),
            "sector": data.get("Sector", ""),
            "industry": data.get("Industry", ""),
            "market_cap": data.get("MarketCapitalization", ""),
            "pe_ratio": data.get("PERatio", ""),
            "peg_ratio": data.get("PEGRatio", ""),
            "price_to_book": data.get("PriceToBookRatio", ""),
            "dividend_yield": data.get("DividendYield", ""),
            "eps": data.get("EPS", ""),
            "revenue_per_share": data.get("RevenuePerShareTTM", ""),
            "profit_margin": data.get("ProfitMargin", ""),
            "operating_margin": data.get("OperatingMarginTTM", ""),
            "return_on_assets": data.get("ReturnOnAssetsTTM", ""),
            "return_on_equity": data.get("ReturnOnEquityTTM", ""),
            "revenue": data.get("RevenueTTM", ""),
            "gross_profit": data.get("GrossProfitTTM", ""),
            "diluted_eps": data.get("DilutedEPSTTM", ""),
            "quarterly_earnings_growth": data.get("QuarterlyEarningsGrowthYOY", ""),
            "quarterly_revenue_growth": data.get("QuarterlyRevenueGrowthYOY", ""),
            "analyst_target_price": data.get("AnalystTargetPrice", ""),
            "trailing_pe": data.get("TrailingPE", ""),
            "forward_pe": data.get("ForwardPE", ""),
            "price_to_sales": data.get("PriceToSalesRatioTTM", ""),
            "beta": data.get("Beta", ""),
            "52_week_high": data.get("52WeekHigh", ""),
            "52_week_low": data.get("52WeekLow", ""),
            "50_day_ma": data.get("50DayMovingAverage", ""),
            "200_day_ma": data.get("200DayMovingAverage", ""),
            "shares_outstanding": data.get("SharesOutstanding", ""),
            "timestamp": datetime.now().isoformat()
        }
        
        return overview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching overview for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search", 
         summary="Search for stock symbols")
def search_symbols(keywords: str = Query(..., min_length=1, description="Search keywords"), 
                  client: AlphaVantageClient = Depends(get_client)):
    """Search for stock symbols by keywords"""
    try:
        data = client.search_symbol(keywords)
        
        if "bestMatches" not in data:
            return {"keywords": keywords, "matches": [], "count": 0}
        
        matches = []
        for match in data["bestMatches"]:
            formatted_match = {
                "symbol": match.get("1. symbol", ""),
                "name": match.get("2. name", ""),
                "type": match.get("3. type", ""),
                "region": match.get("4. region", ""),
                "market_open": match.get("5. marketOpen", ""),
                "market_close": match.get("6. marketClose", ""),
                "timezone": match.get("7. timezone", ""),
                "currency": match.get("8. currency", ""),
                "match_score": match.get("9. matchScore", "")
            }
            matches.append(formatted_match)
        
        return {
            "keywords": keywords,
            "matches": matches,
            "count": len(matches),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching symbols with keywords '{keywords}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Legacy endpoint for backward compatibility
@app.get("/get_stock_data", 
         summary="Legacy endpoint - Get daily stock data",
         deprecated=True)
def get_stock_data(symbol: str, client: AlphaVantageClient = Depends(get_client)):
    """Legacy endpoint for backward compatibility"""
    try:
        data = client.get_time_series(symbol, "TIME_SERIES_DAILY", "compact")
        return data
    except Exception as e:
        logger.error(f"Error in legacy endpoint for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
def health_check():
    """API health check"""
    return {
        "status": "healthy" if alpha_vantage else "degraded",
        "api_key_configured": bool(API_KEY),
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/", summary="API Information")
def root():
    """API root with information"""
    return {
        "name": "Enhanced Stock Data API",
        "version": "2.0.0",
        "description": "Comprehensive stock market data API powered by Alpha Vantage",
        "endpoints": {
            "quote": "/stock/{symbol}/quote",
            "time_series": "/stock/{symbol}/timeseries",
            "overview": "/stock/{symbol}/overview", 
            "search": "/search?keywords={keywords}",
            "legacy": "/get_stock_data?symbol={symbol}"
        },
        "documentation": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
