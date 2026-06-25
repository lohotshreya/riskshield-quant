
import streamlit as st
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import plotly.express as px

from src.utils import fetch_portfolio_data, calculate_returns
from src.engine import HRPEngine
from src.simulator import StressSimulator

st.set_page_config(layout="wide", page_title="RiskShield.quant Engine")

st.title("🛡️ RiskShield.quant")
st.subheader("Asset Allocation via Hierarchical Risk Parity (HRP) & Macro Stress-Testing")
st.markdown("---")

# Sidebar configurations
st.sidebar.header("🕹️ Portfolio Configurations")
ticker_input = st.sidebar.text_input("Tickers (Comma Separated)", "SPY, QQQ, TLT, GLD, USO, VNQ")
tickers = [t.strip().upper() for t in ticker_input.split(",")]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# Execution Trigger
if st.sidebar.button("Run Allocation Engine"):
    with st.spinner("Downloading and processing market structural data..."):
        # Fetch & process
        raw_data = fetch_portfolio_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        returns = calculate_returns(raw_data)
        
        # HRP Execution
        engine = HRPEngine(returns)
        weights = engine.allocate()
        
        # Layout metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📊 Optimal HRP Weight Allocation")
            weight_df = pd.DataFrame({'Asset': weights.index, 'Allocation (%)': (weights.values * 100).round(2)})
            fig_pie = px.pie(weight_df, values='Allocation (%)', names='Asset', hole=0.4,
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(weight_df.set_index('Asset'))

        with col2:
            st.markdown("### 🌲 Structural Asset Cluster Dendrogram")
            fig, ax = plt.subplots(figsize=(6, 4))
            # Generate actual structural tree hierarchy layout
            sch.dendrogram(engine.linkage, labels=engine.tickers, ax=ax, orientation='top')
            plt.title("Asset Risk Clusters")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("Assets linked closely under branches share latent underlying risk structures.")

        st.markdown("---")
        st.markdown("### 🌪️ Institutional Forward-Looking Macro Stress Testing Report")
        
        # Stress testing simulator execution
        simulator = StressSimulator(returns, weights.values)
        stress_report = simulator.run_stress_test()
        
        # Display as clear metric table cards
        st.table(stress_report)
        
        # Benchmark structural analysis
        st.markdown("### 📈 Historical Volatility Profile Comparison")
        portfolio_returns = np.dot(returns, weights.values)
        cum_returns = np.exp(np.cumsum(portfolio_returns)) - 1
        
        # Compare with baseline Equal Weight Allocation (1/N Portfolio)
        eq_weights = np.ones(len(tickers)) / len(tickers)
        eq_returns = np.dot(returns, eq_weights)
        eq_cum_returns = np.exp(np.cumsum(eq_returns)) - 1
        
        performance_df = pd.DataFrame({
            'HRP Strategy Portfolio': cum_returns,
            'Traditional Equal-Weight (1/N)': eq_cum_returns
        }, index=returns.index)
        
        fig_perf = px.line(performance_df, labels={'value': 'Cumulative Performance', 'Date': 'Timeline'})
        st.plotly_chart(fig_perf, use_container_width=True)
