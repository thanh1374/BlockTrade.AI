from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

SEQ_LEN = 50
FEATURES = ["value_eth", "gas_price_gwei", "gas_used_pct", "input_len", "is_contract_call", "failed"]


class MEVDetector:
    def __init__(self, model_path: str = "mev_bot_detector.keras"):

        try:
            # Load and compile model for CPU
            with tf.device("/CPU:0"):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        except Exception as e:
            print(f"Model loading error: {e}")
            self.model = None

    def build_sequence(self, txs: pd.DataFrame) -> np.ndarray:
        """Transform transaction data into sequence for model input."""
        txs = txs.copy()
        txs["time_dt"] = pd.to_datetime(txs["time_utc"], format="%Y%m%d %H%M%S")
        txs = txs.sort_values("time_dt")
        arr = txs[FEATURES].to_numpy()

        if arr.shape[0] < SEQ_LEN:
            pad_len = SEQ_LEN - arr.shape[0]
            pad = np.repeat(arr[:1], pad_len, axis=0)
            seq = np.vstack([pad, arr])
        else:
            seq = arr[-SEQ_LEN:]

        return seq

    def predict(self, transactions: pd.DataFrame) -> Dict:
        """Make prediction using CPU."""
        if self.model is None:
            return {"prob": 0.0, "label": "Model not loaded", "confidence": 0.0}

        with tf.device("/CPU:0"):
            sequence = self.build_sequence(transactions)
            sequence = np.expand_dims(sequence, 0).astype(np.float32)
            prob = float(self.model(sequence, training=False)[0, 0])
        label = "Likely Dodgy" if prob > 0.4 else "Legit"
        return {"prob": prob, "label": label, "confidence": max(prob, 1 - prob)}


def preprocess_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw transactions into format expected by GRU model.

    Args:
        transactions: List of transaction dictionaries from Etherscan

    Returns:
        DataFrame with processed features
    """

    df = transactions.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"].astype(np.int64), unit="s").dt.strftime("%Y%m%d %H%M%S")

    return df.fillna(0)


def demo_gru_detection(raw_transactions: pd.DataFrame) -> Dict:
    """
    Demo function to analyze a wallet for MEV bot behavior.

    Args:
        address: Ethereum wallet address
        transactions_df: DataFrame with wallet transactions

    Returns:
        Dict with MEV analysis results
    """

    transactions = raw_transactions.copy()

    transactions_df = preprocess_transactions(transactions)
    detector = MEVDetector()
    results = detector.predict(transactions_df)
    # Add interpretation
    risk_level = "High" if results["prob"] > 0.7 else "Medium" if results["prob"] > 0.3 else "Low"

    results.update(
        {
            "risk_level": risk_level,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_transactions": len(transactions_df),
        }
    )

    return results
