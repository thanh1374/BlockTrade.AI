import streamlit as st
from requests import get
from matplotlib import pyplot as plt
from datetime import datetime
import requests
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import time
import requests
import json
import google.generativeai as genai
import numpy as np
from app.config import BASE_URL, ETHERSCAN_API_KEY, ETHER_VALUE, url

def make_api_url(module, action, address, **kwargs):
    url = BASE_URL + f"?module={module}&action={action}&address={address}&apikey={ETHERSCAN_API_KEY}"
    for key, value in kwargs.items():
        url += f"&{key}={value}"
    return url

def get_account_balance(address):
    balance_url = make_api_url("account", "balance", address, tag="latest")
    response = get(balance_url)
    data = response.json()
    value = int(data["result"]) / ETHER_VALUE
    return value

def classify_wallet(address):
    known_cex = [
        "0x3f5ce5fbfe3e9af3971d0ef5ccc0c5a6de5d6032",  # Binance Hot
        "0xd551234ae421e3bcba99a0da6d736074f22192ff"   # Binance Cold
    ]
    if address.lower() in known_cex:
        return "CEX"

    code_url = make_api_url("proxy", "eth_getCode", address)
    response = requests.get(code_url)
    code = response.json().get("result", "")
    if code != "0x":
        return "Contract"

    return "Personal"


def get_latest_transactions(address, limit=10):
    latest_tx_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=limit, sort="desc")
    response = requests.get(latest_tx_url)
    transactions = response.json().get("result", [])

    latest_transactions = []
    for tx in transactions:
        tx_data = {
            "hash": tx.get("hash"),
            "from": tx.get("from"),
            "to": tx.get("to"),
            "value_eth": int(tx.get("value", "0")) / ETHER_VALUE,
            "timestamp": datetime.fromtimestamp(int(tx.get("timeStamp"))).strftime("%Y-%m-%d %H:%M:%S"),
            "gas_used": int(tx.get("gasUsed", "0")),
            "gas_price_gwei": int(tx.get("gasPrice", "0")) / (10 ** 9),
            "status": "Success" if tx.get("isError", "0") == "0" else "Failed"
        }
        latest_transactions.append(tx_data)

    return latest_transactions
def get_eth_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data["ethereum"]["usd"]

def plot_balance_over_time(address):
    # Gọi API transaction thường
    transactions_url = make_api_url("account", "txlist", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    data = get(transactions_url).json()["result"]

    # Gọi API internal transaction
    internal_tx_url = make_api_url("account", "txlistinternal", address, startblock=0, endblock=99999999, page=1, offset=10000, sort="asc")
    data2 = get(internal_tx_url).json()["result"]

    data.extend(data2)
    data.sort(key=lambda x: int(x['timeStamp']))

    # Tính toán số dư theo thời gian
    current_balance = 0
    balances = []
    times = []

    for tx in data:
        to = tx.get("to", "").lower()
        from_addr = tx.get("from", "").lower()
        value = int(tx.get("value", 0)) / ETHER_VALUE

        gas = int(tx.get("gasUsed", 0)) * int(tx.get("gasPrice", 0)) / ETHER_VALUE if "gasPrice" in tx else 0
        time = datetime.fromtimestamp(int(tx['timeStamp']))

        if to == address.lower():
            current_balance += value
        elif from_addr == address.lower():
            current_balance -= value + gas

        balances.append(current_balance)
        times.append(time)

    # Tạo biểu đồ với Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=balances,
        mode='lines+markers',
        name='ETH Balance',
        line=dict(color='royalblue', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        xaxis_title='⏱ Time',
        yaxis_title='💰 Balance',
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=20, r=20, t=50, b=30),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def calculate_risk_score(address):
    # Step 1: Gọi API lấy giao dịch
    url = make_api_url(
        module="account",
        action="txlist",
        address=address,
        startblock=0,
        endblock=99999999,
        sort="asc"
    )
    r = requests.get(url)
    data = r.json()

    tx_df = pd.DataFrame(data['result'])
    if tx_df.empty:
        return 0, "Địa chỉ không có giao dịch.", None

    # Step 2: Tiền xử lý
    tx_df['value'] = tx_df['value'].astype(float)
    tx_df['gasPrice'] = tx_df['gasPrice'].astype(float)
    tx_df['isError'] = tx_df['isError'].astype(int)
    tx_df['timeStamp'] = pd.to_datetime(tx_df['timeStamp'], unit='s')
    tx_df = tx_df.sort_values(by='timeStamp')
    tx_df['delta_time'] = tx_df['timeStamp'].diff().dt.total_seconds().fillna(999999)
    tx_df['is_contract'] = tx_df.apply(lambda row: 1 if (row['input'] != '0x' or row['contractAddress']) else 0, axis=1)

    def normalize(series, max_val=None):
        if max_val:
            return np.clip(series / max_val, 0, 1)
        return (series - series.min()) / (series.max() - series.min() + 1e-6)

    tx_df['value_norm'] = normalize(tx_df['value'], max_val=1e22)
    tx_df['gasPrice_norm'] = normalize(tx_df['gasPrice'])
    tx_df['delta_time_norm'] = 1 - normalize(tx_df['delta_time'], max_val=3600)

    w_error = 2
    w_contract = 2
    w_value = 1.5
    w_gas = 1
    w_delta = 2

    tx_df['de_anonymous_score'] = (
        w_error * tx_df['isError'] +
        w_contract * tx_df['is_contract'] +
        w_value * tx_df['value_norm'] +
        w_gas * tx_df['gasPrice_norm'] +
        w_delta * tx_df['delta_time_norm']
    )
    tx_df['de_anonymous_score'] = normalize(tx_df['de_anonymous_score'], max_val=10)
    tx_df['Tx_id'] = tx_df.index

    # Step 3: Tạo thông tin account
    addresses = pd.Series(tx_df['from'].tolist() + tx_df['to'].tolist()).dropna().unique()
    out_degree = tx_df['from'].value_counts()
    in_degree = tx_df['to'].value_counts()

    account_df = pd.DataFrame({'index': addresses})
    account_df['out_degree'] = account_df['index'].map(out_degree).fillna(0).astype(int)
    account_df['in_degree'] = account_df['index'].map(in_degree).fillna(0).astype(int)
    account_df['unsupervised_reliable_0'] = 0.5

    Payer_TxNumber = account_df.set_index('index')['out_degree']
    Payee_TxNumber = account_df.set_index('index')['in_degree']
    nodes = account_df['index']
    Tx_Score = tx_df['de_anonymous_score']
    reliable_0 = account_df.set_index('index')['unsupervised_reliable_0']
    payer = tx_df['from']
    payee = tx_df['to']
    edges = tx_df['Tx_id']

    reliable = {node: reliable_0[node] for node in nodes}
    trust = {node: 0.5 for node in nodes}
    confidence = {edge: 0.5 for edge in edges}

    iter = 0
    trust_value_all = {node: 0 for node in nodes}
    reliable_value_all = {node: 0 for node in nodes}

    while iter < 400:
        d_reliable = d_trust = d_confidence = 0

        for node in nodes:
            trust_value_all[node] = 0
            reliable_value_all[node] = 0

        for edge in edges:
            trust_value_all[payee[edge]] += Tx_Score[edge] * confidence[edge]
            reliable_value_all[payer[edge]] += confidence[edge]

        for node in nodes:
            payee_txn = Payee_TxNumber[node]
            trust_for_node = trust_value_all[node] / payee_txn if payee_txn > 0 else 0.5
            trust_for_node = max(0, min(trust_for_node, 1))
            d_trust += abs(trust[node] - trust_for_node)
            trust[node] = trust_for_node

        for node in nodes:
            payer_txn = Payer_TxNumber[node]
            reliable_for_node = reliable_value_all[node] / payer_txn if payer_txn > 0 else reliable_0[node]
            reliable_for_node = max(0, min(reliable_for_node, 1))
            d_reliable += abs(reliable[node] - reliable_for_node)
            reliable[node] = reliable_for_node

        for edge in edges:
            account_from_reliable = reliable[payer[edge]]
            account_to_trust = trust[payee[edge]]
            transaction_score = Tx_Score[edge]
            x = (account_from_reliable + (1 - abs(transaction_score - account_to_trust))) / 2
            x = max(0, min(x, 1))
            d_confidence += abs(confidence[edge] - x)
            confidence[edge] = x

        iter += 1
        if d_trust < 0.01 and d_confidence < 0.01 and d_reliable < 0.01:
            break

    RISK = {node: 10 - 10 * reliable[node] for node in nodes}
    risk_score = RISK.get(address.lower(), 0)

    result_df = pd.DataFrame({
        'address': list(nodes),
        'trust': [trust[node] for node in nodes],
        'reliable': [reliable[node] for node in nodes],
        'risk_score': [RISK[node] for node in nodes]
    })

    if not result_df.empty:
        fig_risk = px.histogram(result_df, x='risk_score', nbins=20, title='Risk Score Distribution',
                                labels={'risk_score': 'Risk Score'},
                                color_discrete_sequence=['#EF553B'])
        fig_risk.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=30))
    else:
        fig_risk = None

    explanation = f"""
    <div style="border: 1px solid #334155; border-radius:10px; padding: 12px; background-color: #1E293B;">
        <h6><strong>Risk Report</strong></h6>
        <p><strong>Number of Accounts (Nodes):</strong> {len(nodes)}</p>
        <p><strong>Number of Transactions:</strong> {len(edges)}</p>
        <p><strong>Trust Score:</strong> <span style="color:#F59E0B;">{trust.get(address.lower(), 0.5):.4f}</span></p>
        <p><strong>Reliability Score:</strong> <span style="color:#10B981;">{reliable.get(address.lower(), 0.5):.4f}</span></p>
        <p><strong>Iterations Until Convergence:</strong> {iter}</p>
    </div>
    """

    return risk_score, explanation, fig_risk

def explain_wallet_behavior(address, risk_score, wallet_type, txs):
    txs_summary = ""
    for tx in txs[:10]:
        txs_summary += f"- {tx['timestamp']}: {tx['from']} → {tx['to']}, {tx['value_eth']:.4f} ETH, Status: {tx['status']}\n"

    prompt = f"""
You are a blockchain intelligence analyst.

Analyze this Ethereum wallet:
- Address: {address}
- Type: {wallet_type}
- Risk Score: {risk_score}/10
- Recent Transactions:
{txs_summary}

Summarize wallet behavior, potential risks, and red flags in 5-10 sentences.
"""

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"⚠️ Error: {response.status_code} - {response.text}"