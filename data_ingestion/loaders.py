import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class StockDataLoader:
    """Load real-time and historical stock data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price and basic info"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0
                
                return {
                    'symbol': symbol.upper(),
                    'current_price': round(current_price, 2),
                    'previous_close': round(prev_close, 2),
                    'change': round(change, 2),
                    'change_percent': round(change_pct, 2),
                    'volume': hist['Volume'].iloc[-1] if not hist.empty else 0,
                    'market_cap': info.get('marketCap', 'N/A'),
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def get_portfolio_exposure(self, symbols: List[str]) -> Dict[str, Any]:
        """Calculate portfolio exposure for given symbols"""
        try:
            exposures = {}
            total_value = 0
            
            for symbol in symbols:
                data = self.get_stock_price(symbol)
                # Simulate portfolio position (in real app, this would come from database)
                position_value = np.random.uniform(10000, 50000)  # Mock position
                exposures[symbol] = {
                    'value': round(position_value, 2),
                    'price': data['current_price'],
                    'change_percent': data['change_percent']
                }
                total_value += position_value
            
            # Calculate exposure percentages
            for symbol in exposures:
                exposures[symbol]['exposure_percent'] = round(
                    (exposures[symbol]['value'] / total_value) * 100, 2
                )
            
            return {
                'exposures': exposures,
                'total_value': round(total_value, 2),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio exposure: {e}")
            return self._get_fallback_portfolio()
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback data when API fails"""
        fallback_prices = {
            'NVDA': {'price': 875.50, 'change': 12.45},
            'TSMC': {'price': 145.20, 'change': -2.30},
            'AAPL': {'price': 185.75, 'change': 1.85},
            'MSFT': {'price': 420.30, 'change': 5.20}
        }
        
        data = fallback_prices.get(symbol.upper(), {'price': 100.0, 'change': 0.0})
        change_pct = (data['change'] / (data['price'] - data['change'])) * 100
        
        return {
            'symbol': symbol.upper(),
            'current_price': data['price'],
            'previous_close': data['price'] - data['change'],
            'change': data['change'],
            'change_percent': round(change_pct, 2),
            'volume': 1000000,
            'market_cap': 'N/A',
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }
    
    def _get_fallback_portfolio(self) -> Dict[str, Any]:
        """Fallback portfolio data"""
        return {
            'exposures': {
                'NVDA': {'value': 45000, 'exposure_percent': 22.5, 'price': 875.50, 'change_percent': 1.44},
                'TSMC': {'value': 30000, 'exposure_percent': 15.0, 'price': 145.20, 'change_percent': -1.56},
                'AAPL': {'value': 35000, 'exposure_percent': 17.5, 'price': 185.75, 'change_percent': 1.01}
            },
            'total_value': 200000,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

class EarningsDataLoader:
    """Load earnings data and surprises"""
    
    def __init__(self):
        self.earnings_cache = {}
    
    def get_recent_earnings(self, symbols: List[str]) -> Dict[str, Any]:
        """Get recent earnings surprises for given symbols"""
        try:
            earnings_data = {}
            
            for symbol in symbols:
                # In real implementation, this would call earnings API
                # For now, we'll simulate earnings data
                earnings_data[symbol] = self._simulate_earnings_data(symbol)
            
            return {
                'earnings': earnings_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching earnings data: {e}")
            return self._get_fallback_earnings()
    
    def _simulate_earnings_data(self, symbol: str) -> Dict[str, Any]:
        """Simulate earnings data (replace with real API in production)"""
        
        # Sample earnings surprises
        earnings_scenarios = {
            'NVDA': {'eps_actual': 5.16, 'eps_estimate': 4.91, 'surprise': 5.1, 'date': '2024-11-20'},
            'TSMC': {'eps_actual': 1.47, 'eps_estimate': 1.41, 'surprise': 4.3, 'date': '2024-10-17'},
            'AAPL': {'eps_actual': 1.64, 'eps_estimate': 1.60, 'surprise': 2.5, 'date': '2024-11-01'},
            'MSFT': {'eps_actual': 3.30, 'eps_estimate': 3.10, 'surprise': 6.5, 'date': '2024-10-24'}
        }
        
        data = earnings_scenarios.get(symbol.upper(), {
            'eps_actual': 1.0,
            'eps_estimate': 0.95,
            'surprise': 5.3,
            'date': '2024-11-15'
        })
        
        return {
            'symbol': symbol.upper(),
            'earnings_per_share': data['eps_actual'],
            'estimate': data['eps_estimate'],
            'surprise_percent': data['surprise'],
            'beat_estimate': data['eps_actual'] > data['eps_estimate'],
            'report_date': data['date'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_fallback_earnings(self) -> Dict[str, Any]:
        """Fallback earnings data"""
        return {
            'earnings': {
                'TSMC': {
                    'symbol': 'TSMC',
                    'earnings_per_share': 1.47,
                    'estimate': 1.41,
                    'surprise_percent': 4.3,
                    'beat_estimate': True,
                    'report_date': '2024-10-17'
                },
                'NVDA': {
                    'symbol': 'NVDA', 
                    'earnings_per_share': 5.16,
                    'estimate': 4.91,
                    'surprise_percent': 5.1,
                    'beat_estimate': True,
                    'report_date': '2024-11-20'
                }
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

class RiskDataLoader:
    """Load and calculate risk metrics"""
    
    def __init__(self):
        self.stock_loader = StockDataLoader()
        self.earnings_loader = EarningsDataLoader()
    
    def get_asia_tech_exposure(self) -> Dict[str, Any]:
        """Get current Asia tech exposure data"""
        
        asia_tech_symbols = ['TSMC', 'NVDA', '2330.TW', 'ASML']  # Mix of Asia tech
        
        try:
            portfolio_data = self.stock_loader.get_portfolio_exposure(asia_tech_symbols)
            earnings_data = self.earnings_loader.get_recent_earnings(['TSMC', 'NVDA'])
            
            # Calculate total Asia tech exposure
            total_exposure = sum([
                exp['exposure_percent'] 
                for exp in portfolio_data['exposures'].values()
            ])
            
            # Simulate previous day's exposure for comparison
            previous_exposure = total_exposure * np.random.uniform(0.85, 1.15)
            change_percentage = ((total_exposure - previous_exposure) / previous_exposure) * 100
            
            return {
                'asset_class': 'Asia Tech',
                'exposure': {
                    'current': round(total_exposure, 1),
                    'previous': round(previous_exposure, 1),
                    'change_percentage': round(change_percentage, 4),
                    'change_direction': 'increase' if change_percentage > 0 else 'decrease'
                },
                'risk_metrics': {
                    'risk_score': round(total_exposure / 100, 2),  # Simple risk score
                    'risk_level': 'Low' if total_exposure < 30 else 'Medium' if total_exposure < 60 else 'High',
                    'allocation_category': 'Conservative' if total_exposure < 25 else 'Moderate' if total_exposure < 50 else 'Aggressive'
                },
                'holdings': portfolio_data['exposures'],
                'recent_earnings': earnings_data['earnings'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Asia tech exposure: {e}")
            return self._get_fallback_risk_data()
    
    def get_stock_risk_profile(self, symbol: str) -> Dict[str, Any]:
        """Get risk profile for individual stock"""
        try:
            stock_data = self.stock_loader.get_stock_price(symbol)
            
            # Simple risk calculation based on volatility (mock)
            volatility = abs(stock_data['change_percent']) * np.random.uniform(1.5, 3.0)
            
            risk_level = 'Low' if volatility < 15 else 'Medium' if volatility < 30 else 'High'
            
            return {
                'symbol': symbol.upper(),
                'current_price': stock_data['current_price'],
                'daily_change': stock_data['change_percent'],
                'volatility': round(volatility, 2),
                'risk_level': risk_level,
                'market_cap': stock_data.get('market_cap', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk profile for {symbol}: {e}")
            return {
                'symbol': symbol.upper(),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_fallback_risk_data(self) -> Dict[str, Any]:
        """Fallback risk data when calculations fail"""
        return {
            'asset_class': 'Asia Tech',
            'exposure': {
                'current': 22.0,
                'previous': 18.0,
                'change_percentage': 22.2222,
                'change_direction': 'increase'
            },
            'risk_metrics': {
                'risk_score': 0.22,
                'risk_level': 'Low',
                'allocation_category': 'Moderate'
            },
            'holdings': {
                'TSMC': {'value': 45000, 'exposure_percent': 12.0, 'change_percent': 4.3},
                'NVDA': {'value': 40000, 'exposure_percent': 10.0, 'change_percent': 5.1}
            },
            'recent_earnings': {
                'TSMC': {'surprise_percent': 4.3, 'beat_estimate': True},
                'NVDA': {'surprise_percent': 5.1, 'beat_estimate': True}
            },
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback'
        }

class MarketSentimentLoader:
    """Load market sentiment and news data"""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    def get_market_sentiment(self, region: str = 'asia') -> Dict[str, Any]:
        """Get current market sentiment for region"""
        
        # Simulate market sentiment (in real app, would use news/sentiment APIs)
        sentiment_options = ['bullish', 'neutral', 'bearish', 'cautious', 'optimistic']
        current_sentiment = np.random.choice(sentiment_options)
        
        factors = [
            'rising yields',
            'earnings surprises', 
            'geopolitical tensions',
            'central bank policy',
            'tech sector rotation',
            'AI chip demand'
        ]
        
        key_factor = np.random.choice(factors)
        
        return {
            'region': region.title(),
            'sentiment': current_sentiment,
            'confidence': round(np.random.uniform(0.6, 0.9), 2),
            'key_factors': [key_factor],
            'timestamp': datetime.now().isoformat()
        }

# Main data loader class that combines all loaders
class FinancialDataLoader:
    """Main class that combines all data loading functionality"""
    
    def __init__(self):
        self.stock_loader = StockDataLoader()
        self.earnings_loader = EarningsDataLoader()
        self.risk_loader = RiskDataLoader()
        self.sentiment_loader = MarketSentimentLoader()
    
    def get_comprehensive_data(self, query_type: str = 'general') -> Dict[str, Any]:
        """Get comprehensive financial data based on query type"""
        
        try:
            if query_type == 'asia_tech_risk':
                return self.risk_loader.get_asia_tech_exposure()
            
            elif query_type == 'stock_analysis':
                # Default stocks for analysis
                symbols = ['NVDA', 'TSMC', 'AAPL']
                data = {}
                for symbol in symbols:
                    data[symbol] = self.stock_loader.get_stock_price(symbol)
                return data
            
            elif query_type == 'earnings':
                return self.earnings_loader.get_recent_earnings(['NVDA', 'TSMC', 'AAPL'])
            
            elif query_type == 'sentiment':
                return self.sentiment_loader.get_market_sentiment('asia')
            
            else:
                # General comprehensive data
                return {
                    'asia_tech_exposure': self.risk_loader.get_asia_tech_exposure(),
                    'market_sentiment': self.sentiment_loader.get_market_sentiment('asia'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error loading comprehensive data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    loader = FinancialDataLoader()
    
    # Test different data loading scenarios
    print("Asia Tech Risk Data:")
    print(loader.get_comprehensive_data('asia_tech_risk'))
    
    print("\nStock Analysis Data:")
    print(loader.get_comprehensive_data('stock_analysis'))
    
    print("\nEarnings Data:")
    print(loader.get_comprehensive_data('earnings'))