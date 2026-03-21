"""
Volatility targeting for portfolio risk management.
Scales portfolio weights to maintain consistent risk exposure.
"""

import numpy as np
from typing import Optional

class VolatilityScaler:
    """
    Scale portfolio weights based on recent realized volatility.
    
    Target: Maintain consistent volatility exposure across market regimes.
    Method: Scale weights inversely with recent volatility.
    """
    
    def __init__(
        self,
        target_vol: float = 0.15,      # 15% annualized target volatility
        lookback_weeks: int = 20,       # 20-week rolling window
        max_leverage: float = 1.5,      # Max 1.5x leverage
        min_leverage: float = 0.3,      # Min 30% exposure
        weeks_per_year: float = 52.0,
    ):
        self.target_vol = target_vol
        self.lookback_weeks = lookback_weeks
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.weeks_per_year = weeks_per_year
        
        # Buffer to store recent returns
        self.return_buffer = []
    
    def update(self, portfolio_return: float) -> None:
        """Add a new return to the buffer."""
        self.return_buffer.append(portfolio_return)
        if len(self.return_buffer) > self.lookback_weeks:
            self.return_buffer.pop(0)
    
    def get_scale_factor(self) -> float:
        """
        Compute volatility scaling factor.
        
        Returns
        -------
        scale : float
            Multiplier for portfolio weights. 
            > 1.0 when vol is low (increase exposure)
            < 1.0 when vol is high (reduce exposure)
        """
        if len(self.return_buffer) < 10:  # Need minimum history
            return 1.0
        
        # Realized volatility (annualized)
        returns = np.array(self.return_buffer)
        realized_vol = float(np.std(returns, ddof=1) * np.sqrt(self.weeks_per_year))
        
        if realized_vol < 1e-6:  # Avoid division by zero
            return 1.0
        
        # Scale inversely with volatility
        scale = self.target_vol / realized_vol
        
        # Clamp to reasonable bounds
        scale = np.clip(scale, self.min_leverage, self.max_leverage)
        
        return float(scale)
    
    def scale_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply volatility scaling to portfolio weights.
        
        Parameters
        ----------
        weights : np.ndarray [K]
            Raw portfolio weights (sum to 1.0)
        
        Returns
        -------
        scaled_weights : np.ndarray [K]
            Volatility-adjusted weights
        """
        scale = self.get_scale_factor()
        
        # Scale weights
        scaled = weights * scale
        
        # If scale < 1.0, we have cash left over
        # Allocate to cash (represented as reducing all weights proportionally)
        if scale < 1.0:
            # Weights already scaled down, cash is implicit (1 - sum(scaled))
            pass
        
        return scaled
    
    def reset(self) -> None:
        """Clear the return buffer (call at start of new episode)."""
        self.return_buffer = []
