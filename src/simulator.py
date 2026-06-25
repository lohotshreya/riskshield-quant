import numpy as np
import pandas as pd

class StressSimulator:
    def __init__(self, returns_df, weights):
        self.returns = returns_df
        self.weights = np.array(weights)
        self.tickers = returns_df.columns.tolist()

    def calculate_cvar(self, returns, alpha=0.05):
        """Calculates historical Conditional Value at Risk (Expected Shortfall)."""
        portfolio_returns = np.dot(returns, self.weights)
        cutoff = np.quantile(portfolio_returns, alpha)
        cvar = portfolio_returns[portfolio_returns <= cutoff].mean()
        return abs(cvar)

    def simulate_shock(self, scenario):
        """
        Injects engineered structural shocks into asset returns to simulate macro crises.
        """
        shocked_returns = self.returns.copy()
        
        if scenario == "Inflation Spike":
            # Growth/Bonds crash; Commodities/Gold surge
            for col in shocked_returns.columns:
                if col in ['TLT', 'SPY', 'QQQ']:
                    shocked_returns[col] -= 0.025  # Daily penalty shift
                if col in ['GLD', 'USO', 'DBC']:
                    shocked_returns[col] += 0.035  # Daily positive shift
                    
        elif scenario == "Liquidity Crunch (2008 Style)":
            # Systematic correlation breakdown: everything drops significantly except safe havens
            for col in shocked_returns.columns:
                if col != 'TLT':
                    shocked_returns[col] -= 0.040
                else:
                    shocked_returns[col] += 0.010
                    
        elif scenario == "Tech Volatility Shock":
            # Isolated heavy drawdown in tech/growth equities
            for col in shocked_returns.columns:
                if col in ['QQQ', 'AAPL', 'MSFT', 'NVDA']:
                    shocked_returns[col] = shocked_returns[col] * 2.5 - 0.03
                    
        return shocked_returns

    def run_stress_test(self):
        """Runs the benchmark scenario alongside all macro shocks."""
        scenarios = ["Baseline", "Inflation Spike", "Liquidity Crunch (2008 Style)", "Tech Volatility Shock"]
        results = {}
        
        for sc in scenarios:
            if sc == "Baseline":
                ret = self.returns
            else:
                ret = self.simulate_shock(sc)
                
            cvar_val = self.calculate_cvar(ret)
            # Rough daily to annual maximum expected drawdown multiplier approximation
            max_dd_est = cvar_val * np.sqrt(252) * 1.5 
            
            results[sc] = {
                "Daily CVaR (95%)": f"{cvar_val:.2%}",
                "Est. Tail Max Drawdown": f"{min(max_dd_est, 1.0):.2%}"
            }
            
        return pd.DataFrame(results).T
