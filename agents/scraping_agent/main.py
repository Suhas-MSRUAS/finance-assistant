from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from bs4 import BeautifulSoup
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, urlparse
import json
from dataclasses import dataclass
import time
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Scraping Agent", version="2.0.0")

# Rate limiting and caching
REQUEST_CACHE: Dict[str, 'CacheEntry'] = {}
RATE_LIMITS: Dict[str, List[float]] = defaultdict(list)  # domain -> list of timestamps
MAX_REQUESTS_PER_MINUTE = 10
CACHE_DURATION = timedelta(minutes=30)

@dataclass
class CacheEntry:
    data: Any
    timestamp: datetime
    expires_at: datetime

class ScrapingConfig(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5)
    include_financials: bool = True
    include_news: bool = False
    include_analyst_ratings: bool = False
    max_news_items: int = Field(default=5, ge=1, le=20)
    
    @validator('ticker')
    def validate_ticker(cls, v):
        if not re.match(r'^[A-Z]{1,5}$', v.upper()):
            raise ValueError('Ticker must be 1-5 uppercase letters')
        return v.upper()

class EarningsData(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[str] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    revenue: Optional[str] = None
    profit_margin: Optional[float] = None
    news_items: List[Dict[str, str]] = Field(default_factory=list)
    analyst_ratings: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    data_freshness: str = "real-time"

class NewsItem(BaseModel):
    title: str
    link: str
    published: Optional[str] = None
    source: Optional[str] = None

class ScrapingResponse(BaseModel):
    success: bool
    data: Optional[EarningsData] = None
    error: Optional[str] = None
    cached: bool = False
    scrape_time_ms: float

class ScrapingAgent:
    def __init__(self):
        self.session_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        }

    def _generate_cache_key(self, ticker: str, config: ScrapingConfig) -> str:
        """Generate cache key for request"""
        config_str = f"{config.include_financials}-{config.include_news}-{config.include_analyst_ratings}-{config.max_news_items}"
        return hashlib.md5(f"{ticker}-{config_str}".encode()).hexdigest()

    def _is_rate_limited(self, domain: str) -> bool:
        """Check if domain is rate limited"""
        now = time.time()
        # Clean old timestamps (older than 1 minute)
        RATE_LIMITS[domain] = [ts for ts in RATE_LIMITS[domain] if now - ts < 60]
        
        if len(RATE_LIMITS[domain]) >= MAX_REQUESTS_PER_MINUTE:
            return True
        
        RATE_LIMITS[domain].append(now)
        return False

    def _get_cached_data(self, cache_key: str) -> Optional[EarningsData]:
        """Get data from cache if valid"""
        if cache_key in REQUEST_CACHE:
            entry = REQUEST_CACHE[cache_key]
            if datetime.now() < entry.expires_at:
                return entry.data
            else:
                del REQUEST_CACHE[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: EarningsData, duration: timedelta = CACHE_DURATION):
        """Cache data with expiration"""
        REQUEST_CACHE[cache_key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            expires_at=datetime.now() + duration
        )

    async def scrape_yahoo_finance(self, config: ScrapingConfig) -> ScrapingResponse:
        """Scrape Yahoo Finance for earnings data"""
        start_time = time.time()
        cache_key = self._generate_cache_key(config.ticker, config)
        
        # Check cache first
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return ScrapingResponse(
                success=True,
                data=cached_data,
                cached=True,
                scrape_time_ms=(time.time() - start_time) * 1000
            )

        # Check rate limiting
        if self._is_rate_limited("finance.yahoo.com"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded for Yahoo Finance")

        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        try:
            async with aiohttp.ClientSession(
                headers=self.session_headers, 
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=10)
            ) as session:
                earnings_data = EarningsData(ticker=config.ticker)
                
                # Scrape main quote page
                if config.include_financials:
                    await self._scrape_quote_data(session, config.ticker, earnings_data)
                    # Add small delay between requests
                    await asyncio.sleep(0.5)
                
                # Scrape news if requested
                if config.include_news:
                    await self._scrape_news_data(session, config.ticker, earnings_data, config.max_news_items)
                    await asyncio.sleep(0.5)
                
                # Scrape analyst ratings if requested
                if config.include_analyst_ratings:
                    await self._scrape_analyst_data(session, config.ticker, earnings_data)

                # Cache the result
                self._cache_data(cache_key, earnings_data)
                
                return ScrapingResponse(
                    success=True,
                    data=earnings_data,
                    cached=False,
                    scrape_time_ms=(time.time() - start_time) * 1000
                )

        except asyncio.TimeoutError:
            error_msg = f"Request timeout for {config.ticker}"
            logger.error(error_msg)
            return ScrapingResponse(
                success=False,
                error=error_msg,
                cached=False,
                scrape_time_ms=(time.time() - start_time) * 1000
            )
        except aiohttp.ClientError as e:
            error_msg = f"Network error for {config.ticker}: {str(e)}"
            logger.error(error_msg)
            return ScrapingResponse(
                success=False,
                error=error_msg,
                cached=False,
                scrape_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            error_msg = f"Scraping failed for {config.ticker}: {str(e)}"
            logger.error(error_msg)
            return ScrapingResponse(
                success=False,
                error=error_msg,
                cached=False,
                scrape_time_ms=(time.time() - start_time) * 1000
            )

    async def _scrape_quote_data(self, session: aiohttp.ClientSession, ticker: str, earnings_data: EarningsData):
        """Scrape main quote page for financial data"""
        url = f"https://finance.yahoo.com/quote/{ticker}"
        
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch quote page for {ticker}: HTTP {response.status}")
                    return
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract company name - Updated selectors for current Yahoo Finance
                try:
                    name_selectors = [
                        'h1[data-testid="ticker-page-title"]',
                        'h1.yf-3a2v0c',
                        'h1[class*="yf-"]',
                        'h1'
                    ]
                    
                    for selector in name_selectors:
                        name_element = soup.select_one(selector)
                        if name_element:
                            full_text = name_element.get_text().strip()
                            # Remove ticker from company name if present
                            if f"({ticker})" in full_text:
                                earnings_data.company_name = full_text.replace(f"({ticker})", "").strip()
                            else:
                                earnings_data.company_name = full_text
                            break
                except Exception as e:
                    logger.warning(f"Failed to extract company name: {e}")

                # Extract current price - Updated selectors
                try:
                    price_selectors = [
                        'fin-streamer[data-field="regularMarketPrice"]',
                        'span[data-testid="qsp-price"]',
                        'fin-streamer[data-symbol]'
                    ]
                    
                    for selector in price_selectors:
                        price_element = soup.select_one(selector)
                        if price_element:
                            price_text = price_element.get_text().replace(',', '').replace('$', '')
                            try:
                                earnings_data.current_price = float(price_text)
                                break
                            except ValueError:
                                continue
                except Exception as e:
                    logger.warning(f"Failed to extract price: {e}")

                # Extract table data for other metrics
                await self._extract_summary_data(soup, earnings_data)
                
        except Exception as e:
            logger.warning(f"Failed to scrape quote data for {ticker}: {e}")

    async def _extract_summary_data(self, soup: BeautifulSoup, earnings_data: EarningsData):
        """Extract data from summary tables"""
        try:
            # Look for summary table with updated selectors
            tables = soup.find_all('table')
            
            # Also look for div-based layouts which Yahoo Finance sometimes uses
            summary_sections = soup.find_all('div', {'data-testid': re.compile(r'.*summary.*', re.I)})
            
            # Process traditional tables
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()
                        
                        self._map_financial_metric(label, value, earnings_data)
            
            # Process div-based layouts
            for section in summary_sections:
                spans = section.find_all('span')
                for i in range(0, len(spans) - 1, 2):
                    try:
                        label = spans[i].get_text().strip()
                        value = spans[i + 1].get_text().strip()
                        self._map_financial_metric(label, value, earnings_data)
                    except (IndexError, AttributeError):
                        continue
                        
        except Exception as e:
            logger.warning(f"Failed to extract summary data: {e}")

    def _map_financial_metric(self, label: str, value: str, earnings_data: EarningsData):
        """Map financial metric labels to earnings data fields"""
        try:
            label_lower = label.lower()
            
            if any(term in label_lower for term in ['market cap', 'market capitalization']):
                earnings_data.market_cap = value
            elif any(term in label_lower for term in ['pe ratio', 'p/e ratio', 'price/earnings']):
                try:
                    clean_value = re.sub(r'[^\d.-]', '', value)
                    if clean_value and clean_value not in ['N/A', '--']:
                        earnings_data.pe_ratio = float(clean_value)
                except (ValueError, TypeError):
                    pass
            elif 'eps' in label_lower or 'earnings per share' in label_lower:
                try:
                    clean_value = re.sub(r'[^\d.-]', '', value)
                    if clean_value and clean_value not in ['N/A', '--']:
                        earnings_data.eps = float(clean_value)
                except (ValueError, TypeError):
                    pass
            elif 'revenue' in label_lower and 'ttm' in label_lower:
                earnings_data.revenue = value
            elif any(term in label_lower for term in ['profit margin', 'net margin']):
                try:
                    clean_value = re.sub(r'[^\d.-]', '', value.replace('%', ''))
                    if clean_value and clean_value not in ['N/A', '--']:
                        earnings_data.profit_margin = float(clean_value)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            logger.warning(f"Failed to map metric {label}: {e}")

    async def _scrape_news_data(self, session: aiohttp.ClientSession, ticker: str, earnings_data: EarningsData, max_items: int):
        """Scrape news data for the ticker"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch news page for {ticker}: HTTP {response.status}")
                    return
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find news items with updated selectors
                news_items = []
                
                # Try multiple selectors for news articles
                article_selectors = [
                    'h3 a',
                    'h4 a', 
                    '[data-testid*="news"] a',
                    '.js-stream-content a',
                    'li[data-testid*="news-item"] a'
                ]
                
                for selector in article_selectors:
                    articles = soup.select(selector)
                    if articles:
                        break
                
                for article in articles[:max_items]:
                    try:
                        title = article.get_text().strip()
                        link = article.get('href', '')
                        
                        if not title or not link:
                            continue
                            
                        if link.startswith('/'):
                            link = urljoin('https://finance.yahoo.com', link)
                        elif not link.startswith('http'):
                            continue
                        
                        news_items.append({
                            'title': title,
                            'link': link,
                            'source': 'Yahoo Finance'
                        })
                        
                        if len(news_items) >= max_items:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse news item: {e}")
                        continue
                
                earnings_data.news_items = news_items
                
        except Exception as e:
            logger.warning(f"Failed to scrape news data for {ticker}: {e}")

    async def _scrape_analyst_data(self, session: aiohttp.ClientSession, ticker: str, earnings_data: EarningsData):
        """Scrape analyst ratings and recommendations"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/analysis"
            
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch analysis page for {ticker}: HTTP {response.status}")
                    return
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract analyst recommendations with updated selectors
                analyst_data = {}
                
                # Look for recommendation tables
                tables = soup.find_all('table')
                for table in tables:
                    # Check table headers or nearby text for recommendation indicators
                    table_text = table.get_text().lower()
                    if any(term in table_text for term in ['recommendation', 'rating', 'analyst']):
                        rows = table.find_all('tr')
                        
                        # Skip header row
                        for row in rows[1:]:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                try:
                                    period = cells[0].get_text().strip()
                                    rating = cells[1].get_text().strip()
                                    if period and rating:
                                        analyst_data[period] = rating
                                except Exception:
                                    continue
                
                earnings_data.analyst_ratings = analyst_data
                
        except Exception as e:
            logger.warning(f"Failed to scrape analyst data for {ticker}: {e}")

# Global scraping agent
scraper = ScrapingAgent()

@app.get("/scrape_earnings", response_model=ScrapingResponse)
async def scrape_earnings(
    ticker: str = Query(..., description="Stock ticker symbol (e.g., AAPL)", regex=r"^[A-Za-z]{1,5}$"),
    include_financials: bool = Query(True, description="Include financial metrics"),
    include_news: bool = Query(False, description="Include recent news"),
    include_analyst_ratings: bool = Query(False, description="Include analyst ratings"),
    max_news_items: int = Query(5, ge=1, le=20, description="Maximum news items to fetch")
):
    """Scrape earnings and financial data for a given ticker"""
    try:
        config = ScrapingConfig(
            ticker=ticker.upper(),
            include_financials=include_financials,
            include_news=include_news,
            include_analyst_ratings=include_analyst_ratings,
            max_news_items=max_news_items
        )
        
        return await scraper.scrape_yahoo_finance(config)
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid ticker format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/scrape_earnings", response_model=ScrapingResponse)
async def scrape_earnings_post(config: ScrapingConfig):
    """Scrape earnings data with detailed configuration"""
    try:
        config.ticker = config.ticker.upper()
        return await scraper.scrape_yahoo_finance(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/scrape_multiple")
async def scrape_multiple_tickers(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    include_financials: bool = Query(True),
    background_tasks: BackgroundTasks = None
):
    """Scrape multiple tickers (use with caution for rate limiting)"""
    try:
        ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        
        if len(ticker_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 tickers allowed")
        
        if not ticker_list:
            raise HTTPException(status_code=400, detail="No valid tickers provided")
        
        results = {}
        
        for ticker in ticker_list:
            if not re.match(r'^[A-Z]{1,5}$', ticker):
                results[ticker] = ScrapingResponse(
                    success=False,
                    error=f"Invalid ticker format: {ticker}",
                    cached=False,
                    scrape_time_ms=0
                )
                continue
                
            try:
                config = ScrapingConfig(
                    ticker=ticker,
                    include_financials=include_financials,
                    include_news=False,  # Disable news for batch to avoid rate limits
                    include_analyst_ratings=False
                )
                
                result = await scraper.scrape_yahoo_finance(config)
                results[ticker] = result
                
                # Add delay between requests to be respectful
                await asyncio.sleep(1.5)
                
            except Exception as e:
                results[ticker] = ScrapingResponse(
                    success=False,
                    error=str(e),
                    cached=False,
                    scrape_time_ms=0
                )
        
        return {"results": results, "total_processed": len(ticker_list)}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/cache_stats")
async def get_cache_stats():
    """Get cache statistics"""
    now = datetime.now()
    active_entries = sum(1 for entry in REQUEST_CACHE.values() if entry.expires_at > now)
    expired_entries = len(REQUEST_CACHE) - active_entries
    
    return {
        "total_cached_entries": len(REQUEST_CACHE),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "cache_hit_potential": f"{(active_entries / max(len(REQUEST_CACHE), 1)) * 100:.1f}%",
        "rate_limit_domains": len(RATE_LIMITS)
    }

@app.delete("/cache")
async def clear_cache():
    """Clear all cached data"""
    cleared_count = len(REQUEST_CACHE)
    REQUEST_CACHE.clear()
    RATE_LIMITS.clear()
    return {
        "status": "cache_cleared", 
        "cleared_entries": cleared_count,
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    now = datetime.now()
    active_cache_entries = sum(1 for entry in REQUEST_CACHE.values() if entry.expires_at > now)
    
    return {
        "status": "healthy",
        "timestamp": now,
        "active_cache_entries": active_cache_entries,
        "total_cache_entries": len(REQUEST_CACHE),
        "rate_limit_domains": len(RATE_LIMITS),
        "version": "2.0.0"
    }

# Startup event to log application start
@app.on_event("startup")
async def startup_event():
    logger.info("Enhanced Scraping Agent started successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")