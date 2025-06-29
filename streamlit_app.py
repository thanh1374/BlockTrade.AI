"""
BlockTrade AI - Ethereum Wallet Analysis Tool
Copyright (c) 2025 TRADE.AI.NHOT
Licensed under the MIT License - see LICENSE file for details
"""
import json
import pandas as pd
import streamlit as st

from app.main import (
    get_account_balance,
    get_latest_transactions,
    classify_wallet,
    get_eth_price,
    plot_balance_over_time,
    calculate_risk_score,
    explain_wallet_behavior,
    display_graph_pyvis,
)
from app.visualize import visualize_wallet
from model import demo_gru_detection


# ========== Page Config ==========
st.set_page_config(page_title="BlockTrace AI", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

# ========== Custom CSS ==========
st.markdown(
    """<style>
:root {
    --primary-dark: #0D1117;
    --secondary-dark: #161B22;
    --accent-blue: #3B82F6;
    --accent-teal: #06B6D4;
    --light-text: #F1F5F9;
    --border-color: #30363D;
    --metric-bg: #1E2530;
}
html, body, [class*="css"] { color: var(--light-text); }
.stApp {
    background: linear-gradient(160deg, var(--primary-dark) 30%, #0F172A 100%);
}
.title {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color);
}
.stRadio > div { flex-direction: column; }
.stRadio > div > label {
    background-color: var(--primary-dark);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    cursor: pointer;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}
.stRadio > div > label:hover {
    background-color: var(--accent-blue);
    color: white !important;
}
.stRadio > div > label[data-selected="true"] {
    background-color: var(--accent-teal);
    color: white !important;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    background-color: var(--secondary-dark);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}
.summary-item {
    display: flex;
    flex-direction: column;
    padding: 0.5rem;
    background-color: var(--metric-bg);
    border-radius: 8px;
}
.summary-key {
    font-weight: 500;
    font-size: 0.95rem;
    color: var(--accent-teal);
    margin-bottom: 4px;
}
.summary-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--light-text);
    word-break: break-word;
}
.sub-value {
    font-size: 0.9rem;
    color: #CBD5E1;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.15);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(6, 182, 212, 0.3);
    background: linear-gradient(135deg, var(--accent-teal), var(--accent-blue));
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: none;
    opacity: 0.95;
}
</style>
""",
    unsafe_allow_html=True,
)

# ========== Header ==========
st.markdown("<div class='title'>💎 BlockTrace AI</div>", unsafe_allow_html=True)

# ========== Input Section ==========
col1, col2 = st.columns([4, 1])
address = col1.text_input("**🔗 Enter Ethereum Address**", placeholder="0x...", label_visibility="collapsed")
analyze_btn = col2.button("Analyze Wallet", use_container_width=True)

if analyze_btn and address:
    st.session_state["wallet_analyzed"] = True
    st.session_state["address"] = address

# ========== Main Section ==========
if st.session_state.get("wallet_analyzed") and st.session_state.get("address"):
    address = st.session_state["address"]
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("#### Select View")
        tab = st.radio(
            label="Menu",
            options=[" Wallet Overview", " Risk Assessment", " Wallet Graph"],
            label_visibility="collapsed",
        )

    with col2:
        if tab == " Wallet Overview":
            st.subheader(" Wallet Overview")
            with st.spinner("Loading wallet info..."):
                balance = get_account_balance(address)
                wallet_type = classify_wallet(address)
                eth_price = get_eth_price()

            st.markdown(
                f"""
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-key">Address</span>
                    <span class="summary-value">{address[:15]}...{address[-4:]}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-key">Wallet Type</span>
                    <span class="summary-value">{wallet_type}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-key">ETH Balance</span>
                    <span class="summary-value">{balance:.4f} <br><span class="sub-value">≈ {balance * eth_price:,.2f} USD</span></span>
                </div>
                <div class="summary-item">
                    <span class="summary-key">ETH Price</span>
                    <span class="summary-value">{eth_price:,.2f} USD</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Portfolio Trend")
            with st.spinner("Please wait..."):
                plot_balance_over_time(address)

            st.markdown("#### Recent Transactions")
            txs = get_latest_transactions(address, 10)
            if not txs:
                st.info("No transactions found.")
            else:
                for i, tx in enumerate(txs):
                    status_color = "#27AE60" if tx.get("status") == "Success" else "#E74C3C"
                    with st.expander(f"🔹 Tx #{i+1} | {tx.get('value_eth', 0):.4f} ETH | Status: {tx.get('status')}"):
                        st.markdown(
                            f"""
                        <div style='background-color: var(--primary-color); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-color);'>
                            <div><b>Timestamp:</b> {tx.get('timestamp')}</div>
                            <div><b>Hash:</b> <code>{tx.get('hash')}</code></div>
                            <div><b>From:</b> {tx.get('from')}</div>
                            <div><b>To:</b> {tx.get('to')}</div>
                            <div><b>Value:</b> {tx.get('value_eth', 0):.4f} ETH (${tx.get('value_usd', 0):,.2f})</div>
                            <div><b>Gas Used:</b> {tx.get('gas_used')}</div>
                            <div><b>Gas Price:</b> {tx.get('gas_price_gwei', 0):.2f} GWei</div>
                            <div><b>Status:</b> <span style='color:{status_color}'>{tx.get('status')}</span></div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        elif tab == " Risk Assessment":
            st.subheader(" Risk Assessment")
            model_type = st.selectbox(
                "Select Analysis Model", options=["Risk Score Analysis", "GRU Detection"], key="model_selector"
            )
            try:
                with st.spinner("Analyzing risk..."):
                    if model_type == "Risk Score Analysis":
                        risk_score, explanation, fig_risk = calculate_risk_score(address)
                        wallet_type = classify_wallet(address)
                        txs = get_latest_transactions(address, 10)

                        level, color = (
                            ("Low", "#10B981")
                            if risk_score <= 4
                            else (("Moderate", "#F59E0B") if risk_score <= 7 else ("High", "#DC2626"))
                        )

                        colA, colB = st.columns([1, 3])
                        with colA:
                            st.markdown(
                                f"""
                            <div style="background-color: #1E293B; padding: 20px; border-radius: 10px; border: 1px solid #334155; font-family: 'Segoe UI', sans-serif;">
                                <h4 style="font-size: 20px; color: #60A5FA; font-weight: 600; margin-bottom: 12px;">Threat Level</h4>
                                <p style="font-size: 28px; font-weight: 700; color: {color}; margin: 0 0 8px 0;">{risk_score:.1f}/10</p>
                                <p style="font-size: 16px; color: #E2E8F0; margin-bottom: 16px;">Level: <strong style="color: #FACC15;">{level}</strong></p>
                                <div style="height: 12px; background-color: #334155; border-radius: 6px; overflow: hidden;">
                                    <div style="width: {risk_score * 10}%; height: 100%; background: linear-gradient(to right, {color}, #FACC15); transition: width 0.6s ease;"></div>
                            """,
                                unsafe_allow_html=True,
                            )

                        with colB:
                            st.markdown(explanation, unsafe_allow_html=True)

                        st.plotly_chart(fig_risk, use_container_width=True)
                    elif model_type == "GRU Detection":
                        txs = get_latest_transactions(address, 50)
                        if not txs:
                            st.warning("No transactions found for analysis")
                        else:
                            df_transactions = pd.DataFrame(txs)
                            gru_results = demo_gru_detection(df_transactions)
                            wallet_type = classify_wallet(address)
                            # Display MEV analysis results
                            prob = gru_results["prob"]
                            risk_color = "#DC2626" if prob > 0.7 else "#F59E0B" if prob > 0.3 else "#10B981"
                            risk_score = prob * 10
                            st.markdown(
                                f"""
                            <div class="analysis-container">
                                <h3>Address Analysis Results</h3>
                                <p><strong style="color: #FFFFFF;">Classification:</strong> <span style="color: #F8FAFC;">{gru_results["label"]}</span></p>
                                <p><strong style="color: #FFFFFF;">Confidence:</strong> <span style="color: #34D399;">{gru_results["confidence"]:.2%}</span></p>
                                <p><strong style="color: #FFFFFF;">Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{gru_results["risk_level"]}</span></p>
                                <p><strong style="color: #FFFFFF;">Transactions Analyzed:</strong> <span style="color: #F8FAFC;">{gru_results["num_transactions"]}</span></p>
                                <div class="analysis-footer">
                                    <span style="color: #94A3B8;">Analysis Time: {gru_results["analysis_time"]}</span>
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    else:
                        st.error("Please select a valid analysis model.")
                st.markdown(
                    """
                    <style>
                    .gradient-header {
                        font-size: 1.8rem;
                        font-weight: 600;
                        margin-bottom: 1rem;
                        background: linear-gradient(90deg, #3B82F6, #06B6D4);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        display: inline-block;
                    }
                    .analysis-container {
                        background-color: #1E293B;
                        padding: 1.5rem;
                        border-radius: 12px;
                        border: 1px solid #334155;
                        color: #F8FAFC;  /* Very light gray for main text */
                        font-size: 1.1rem;
                        line-height: 1.8;
                    }
                    
                    .analysis-container h3 {
                        color: #60A5FA;  /* Light blue for headings */
                        margin-bottom: 1rem;
                        font-size: 1.3rem;
                    }
                    
                    .analysis-container p {
                        margin-bottom: 1rem;
                        color: #E2E8F0;  /* Light gray for better readability */
                    }
                    
                    .analysis-container strong {
                        color: #38BDF8;  /* Bright blue for emphasis */
                        font-weight: 600;
                    }
                    
                    .analysis-container .metric {
                        color: #34D399;  /* Mint green for numbers */
                        font-weight: 600;
                    }
                    
                    .analysis-container .warning {
                        color: #FBBF24;  /* Amber for warnings */
                        font-weight: 600;
                    }
                    
                    .analysis-footer {
                        margin-top: 1rem;
                        font-size: 0.9rem;
                        color: #94A3B8;  /* Subtle gray for footer */
                        border-top: 1px solid #334155;
                        padding-top: 0.5rem;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Evaluation Explain")
                ai_explanation = explain_wallet_behavior(address, risk_score, wallet_type, txs)

                st.markdown(
                    f"""
                    <div class="analysis-container">
                        {ai_explanation}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                st.error(f" Error: {e}")

        elif tab == " Wallet Graph":
            st.subheader(" Wallet Transaction Graph")

            depth = st.number_input(
                " Enter Graph Depth",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="How many levels of connections to expand from the wallet.",
            )

            generate = st.button("Generate Graph", use_container_width=True)

            if generate:
                with st.spinner(" Generating wallet graph..."):
                    success, message = visualize_wallet(address, depth=int(depth))

                if success:
                    st.success(message)

                    try:
                        with open("app/libs/graph.json.js", "r", encoding="utf-8") as f:
                            json_content = f.read().replace("var rendru = ", "").strip().rstrip(";")
                            graph_data = json.loads(json_content)

                        st.markdown(
                            """
                            <div style='text-align: center; margin-top: 10px;'>
                                <span style='color: #EF4444; font-weight: bold;'>🔴 Red: Outgoing (Sent)</span> &nbsp;|&nbsp;
                                <span style='color: #10B981; font-weight: bold;'>🟢 Green: Incoming (Received)</span> 
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )

                        display_graph_pyvis(graph_data, center_wallet=address)

                    except Exception as e:
                        st.error(f"❌ Could not load graph viewer: {e}")
                else:
                    st.error(message)

# ========== Footer ==========
st.markdown("---")
st.caption("© 2025 BlockTrade AI | v1.0.0 | Built by DSEB64A.NHOT")
