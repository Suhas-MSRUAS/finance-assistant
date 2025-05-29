from fastapi import FastAPI, Body, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime, date
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Exposure Analysis API",
    description="Advanced portfolio risk exposure computation and analysis",
    version="2.0.0"
)

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error",
            "errors": exc.errors(),
            "message": "Request validation failed. Check required fields and data types."
        }
    )

# Pydantic models for request validation
class AllocationData(BaseModel):
    today: float = Field(..., ge=0, le=100, description="Current allocation percentage")
    yesterday: float = Field(..., ge=0, le=100, description="Previous allocation percentage")
    asset_class: str = Field(..., description="Asset class name (e.g., 'Asia Tech', 'US Bonds')")
    volatility: Optional[float] = Field(None, ge=0, description="Asset volatility (annualized %)")
    correlation: Optional[float] = Field(None, ge=-1, le=1, description="Correlation with benchmark")
    
    @field_validator('today', 'yesterday')
    @classmethod
    def validate_percentages(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Allocation must be between 0 and 100')
        return v

class PortfolioRiskRequest(BaseModel):
    allocations: List[AllocationData] = Field(..., min_length=1, description="List of asset allocations")
    portfolio_value: Optional[float] = Field(None, gt=0, description="Total portfolio value")
    risk_free_rate: Optional[float] = Field(3.0, ge=0, description="Risk-free rate for calculations")
    confidence_level: Optional[float] = Field(0.95, gt=0, lt=1, description="Confidence level for VaR")

class SingleAssetRequest(BaseModel):
    today: float = Field(..., ge=0, le=100)
    yesterday: float = Field(..., ge=0, le=100)
    asset_class: str = Field(default="Unknown Asset", min_length=1)
    volatility: Optional[float] = Field(None, ge=0)
    
    @field_validator('today', 'yesterday')
    @classmethod
    def validate_percentages(cls, v):
        if v < 0 or v > 100:
            raise ValueError('Allocation must be between 0 and 100')
        return v

# Risk calculation utilities
class RiskCalculator:
    @staticmethod
    def calculate_percentage_change(current: float, previous: float) -> float:
        """Calculate percentage change with zero division protection"""
        if previous == 0:
            return float('inf') if current > 0 else 0.0
        return round(((current - previous) / previous) * 100, 4)
    
    @staticmethod
    def calculate_allocation_risk_score(allocation: float, volatility: Optional[float] = None) -> float:
        """Calculate risk score based on allocation and volatility"""
        base_risk = allocation / 100  # Normalize to 0-1
        if volatility:
            # Higher volatility increases risk exponentially
            volatility_multiplier = 1 + (volatility / 100) ** 1.5
            return round(base_risk * volatility_multiplier, 4)
        return round(base_risk, 4)
    
    @staticmethod
    def calculate_concentration_risk(allocations: List[float]) -> Dict[str, float]:
        """Calculate concentration risk metrics"""
        total_allocation = sum(allocations)
        if total_allocation == 0:
            return {"herfindahl_index": 0, "concentration_ratio": 0, "diversification_ratio": 1}
        
        # Normalize allocations
        normalized = [a / total_allocation for a in allocations]
        
        # Herfindahl-Hirschman Index
        hhi = sum(w**2 for w in normalized)
        
        # Top 3 concentration ratio
        top_3 = sum(sorted(normalized, reverse=True)[:3])
        
        # Diversification ratio (inverse of HHI)
        diversification = 1 / hhi if hhi > 0 else 1
        
        return {
            "herfindahl_index": round(hhi, 4),
            "concentration_ratio": round(top_3, 4),
            "diversification_ratio": round(diversification, 4)
        }
    
    @staticmethod
    def calculate_var(allocation: float, volatility: float, confidence_level: float = 0.95, 
                     portfolio_value: Optional[float] = None) -> Dict[str, float]:
        """Calculate Value at Risk"""
        if volatility <= 0:
            return {"var_percentage": 0, "var_absolute": 0}
        
        # Z-score for confidence level
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z_score = z_scores.get(confidence_level, 1.645)
        
        # Daily VaR calculation
        daily_volatility = volatility / np.sqrt(252)  # Assuming 252 trading days
        var_percentage = z_score * daily_volatility * (allocation / 100)
        
        var_absolute = var_percentage * portfolio_value / 100 if portfolio_value else 0
        
        return {
            "var_percentage": round(var_percentage, 4),
            "var_absolute": round(var_absolute, 2) if portfolio_value else None
        }

calculator = RiskCalculator()

@app.post("/risk_exposure", 
          summary="Compute single asset risk exposure",
          description="Calculate risk metrics for a single asset allocation change")
def compute_risk(data: SingleAssetRequest):
    """Enhanced single asset risk computation"""
    try:
        logger.info(f"Received request data: {data}")
        logger.info(f"Asset class: {data.asset_class}, Today: {data.today}, Yesterday: {data.yesterday}")
        # Basic calculations
        change = calculator.calculate_percentage_change(data.today, data.yesterday)
        risk_score = calculator.calculate_allocation_risk_score(data.today, data.volatility)
        
        # Risk level classification
        risk_level = "Low"
        if data.today > 30:
            risk_level = "Medium"
        if data.today > 50:
            risk_level = "High"
        if data.today > 70:
            risk_level = "Very High"
        
        response = {
            "asset_class": data.asset_class,
            "exposure": {
                "current": data.today,
                "previous": data.yesterday,
                "change_percentage": change,
                "change_direction": "increase" if change > 0 else "decrease" if change < 0 else "unchanged"
            },
            "risk_metrics": {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "allocation_category": "Concentrated" if data.today > 25 else "Moderate" if data.today > 10 else "Minimal"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add volatility-based metrics if available
        if data.volatility:
            var_metrics = calculator.calculate_var(data.today, data.volatility)
            response["risk_metrics"]["value_at_risk"] = var_metrics
            response["risk_metrics"]["volatility_adjusted_exposure"] = round(
                data.today * (data.volatility / 20), 2  # Normalized to 20% baseline volatility
            )
        
        logger.info(f"Risk calculation completed for {data.asset_class}")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating risk: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")

@app.post("/portfolio_risk", 
          summary="Comprehensive portfolio risk analysis",
          description="Calculate comprehensive risk metrics for entire portfolio")
def compute_portfolio_risk(data: PortfolioRiskRequest):
    """Comprehensive portfolio risk analysis"""
    try:
        results = {
            "portfolio_summary": {
                "total_assets": len(data.allocations),
                "total_current_allocation": sum(a.today for a in data.allocations),
                "total_previous_allocation": sum(a.yesterday for a in data.allocations),
                "portfolio_value": data.portfolio_value
            },
            "individual_assets": [],
            "portfolio_metrics": {},
            "risk_warnings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Process each asset
        total_var = 0
        current_allocations = []
        
        for allocation in data.allocations:
            asset_result = {
                "asset_class": allocation.asset_class,
                "current_allocation": allocation.today,
                "previous_allocation": allocation.yesterday,
                "change_percentage": calculator.calculate_percentage_change(allocation.today, allocation.yesterday),
                "risk_score": calculator.calculate_allocation_risk_score(allocation.today, allocation.volatility)
            }
            
            # Add volatility metrics if available
            if allocation.volatility:
                var_metrics = calculator.calculate_var(
                    allocation.today, allocation.volatility, 
                    data.confidence_level, data.portfolio_value
                )
                asset_result["value_at_risk"] = var_metrics
                if var_metrics["var_percentage"]:
                    total_var += var_metrics["var_percentage"]
            
            results["individual_assets"].append(asset_result)
            current_allocations.append(allocation.today)
        
        # Portfolio-level risk metrics
        concentration_metrics = calculator.calculate_concentration_risk(current_allocations)
        results["portfolio_metrics"] = {
            "concentration_risk": concentration_metrics,
            "total_var_percentage": round(total_var, 4) if total_var > 0 else None,
            "average_allocation": round(np.mean(current_allocations), 2),
            "allocation_std_dev": round(np.std(current_allocations), 2),
            "max_single_exposure": max(current_allocations),
            "min_single_exposure": min(current_allocations)
        }
        
        # Risk warnings
        if max(current_allocations) > 40:
            results["risk_warnings"].append("High concentration risk: Single asset exceeds 40%")
        
        if concentration_metrics["herfindahl_index"] > 0.25:
            results["risk_warnings"].append("Portfolio highly concentrated (HHI > 0.25)")
        
        if results["portfolio_summary"]["total_current_allocation"] > 100:
            results["risk_warnings"].append("Total allocation exceeds 100%")
        
        logger.info(f"Portfolio risk analysis completed for {len(data.allocations)} assets")
        return results
        
    except Exception as e:
        logger.error(f"Error in portfolio risk calculation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Portfolio risk calculation failed: {str(e)}")

@app.get("/risk_thresholds")
def get_risk_thresholds():
    """Get recommended risk thresholds and guidelines"""
    return {
        "allocation_thresholds": {
            "minimal": {"min": 0, "max": 10, "description": "Low impact on portfolio"},
            "moderate": {"min": 10, "max": 25, "description": "Reasonable diversified exposure"},
            "concentrated": {"min": 25, "max": 50, "description": "Significant exposure, monitor closely"},
            "high_risk": {"min": 50, "max": 100, "description": "Dominant position, high concentration risk"}
        },
        "concentration_risk": {
            "herfindahl_index": {
                "diversified": {"max": 0.15, "description": "Well diversified portfolio"},
                "moderate": {"min": 0.15, "max": 0.25, "description": "Moderately concentrated"},
                "concentrated": {"min": 0.25, "description": "Highly concentrated portfolio"}
            }
        },
        "volatility_bands": {
            "low": {"max": 10, "description": "Conservative assets"},
            "medium": {"min": 10, "max": 20, "description": "Moderate risk assets"},
            "high": {"min": 20, "max": 35, "description": "Growth/aggressive assets"},
            "very_high": {"min": 35, "description": "Speculative/high-risk assets"}
        }
    }

@app.get("/health")
def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)