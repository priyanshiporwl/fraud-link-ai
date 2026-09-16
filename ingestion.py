"""
ingestion.py
Step 1: Data Ingestion.

Reads fragmented evidence files (CDR/IPDR, bank/UPI transactions, chat/email
logs, APK/system logs) in CSV, Excel, JSON or TXT format, and standardizes
them into a small set of common pandas DataFrames the rest of the pipeline
understands.
"""

import os
import json
import pandas as pd

from utils import (
    register_evidence_file,
    normalize_phone,
    normalize_upi,
    normalize_account,
    normalize_ip,
    normalize_imei,
)

# Canonical column names each record type is normalized to.
TRANSACTION_COLUMNS = [
    "txn_id", "timestamp", "sender_account", "receiver_account",
    "sender_upi", "receiver_upi", "amount", "channel", "source_file",
]

CALL_LOG_COLUMNS = [
    "record_id", "timestamp", "caller_number", "callee_number",
    "imei", "imsi", "tower_ip", "duration_sec", "source_file",
]

CHAT_LOG_COLUMNS = [
    "message_id", "timestamp", "sender", "receiver", "platform",
    "content", "source_file",
]


def _read_any(filepath: str) -> pd.DataFrame:
    """Read csv/xlsx/json/txt into a DataFrame with best-effort parsing."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv" or ext == ".txt":
        df = pd.read_csv(filepath, sep=None, engine="python")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _first_present(df: pd.DataFrame, candidates):
    """Return the first column name from `candidates` that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_transactions(filepath: str, verify_integrity: bool = True) -> pd.DataFrame:
    """Load a bank/UPI transaction file and normalize to TRANSACTION_COLUMNS."""
    if verify_integrity:
        register_evidence_file(filepath)
    raw = _read_any(filepath)

    col_map = {
        "txn_id": _first_present(raw, ["txn_id", "transaction_id", "utr", "ref_no"]),
        "timestamp": _first_present(raw, ["timestamp", "date", "txn_date", "datetime"]),
        "sender_account": _first_present(raw, ["sender_account", "from_account", "debit_account"]),
        "receiver_account": _first_present(raw, ["receiver_account", "to_account", "credit_account", "beneficiary_account"]),
        "sender_upi": _first_present(raw, ["sender_upi", "from_upi", "payer_vpa"]),
        "receiver_upi": _first_present(raw, ["receiver_upi", "to_upi", "payee_vpa"]),
        "amount": _first_present(raw, ["amount", "txn_amount", "value"]),
        "channel": _first_present(raw, ["channel", "mode", "payment_mode"]),
    }

    out = pd.DataFrame()
    for canon, src in col_map.items():
        out[canon] = raw[src] if src else None

    out["sender_account"] = out["sender_account"].map(normalize_account)
    out["receiver_account"] = out["receiver_account"].map(normalize_account)
    out["sender_upi"] = out["sender_upi"].map(normalize_upi)
    out["receiver_upi"] = out["receiver_upi"].map(normalize_upi)
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").fillna(0.0)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["source_file"] = os.path.basename(filepath)
    out = out.dropna(subset=["timestamp"]).drop_duplicates()
    return out


def load_call_logs(filepath: str, verify_integrity: bool = True) -> pd.DataFrame:
    """Load a CDR/IPDR file and normalize to CALL_LOG_COLUMNS."""
    if verify_integrity:
        register_evidence_file(filepath)
    raw = _read_any(filepath)

    col_map = {
        "record_id": _first_present(raw, ["record_id", "cdr_id", "id"]),
        "timestamp": _first_present(raw, ["timestamp", "call_time", "date"]),
        "caller_number": _first_present(raw, ["caller_number", "a_party", "from_number", "msisdn"]),
        "callee_number": _first_present(raw, ["callee_number", "b_party", "to_number"]),
        "imei": _first_present(raw, ["imei"]),
        "imsi": _first_present(raw, ["imsi"]),
        "tower_ip": _first_present(raw, ["tower_ip", "ip_address", "ip"]),
        "duration_sec": _first_present(raw, ["duration_sec", "duration", "call_duration"]),
    }

    out = pd.DataFrame()
    for canon, src in col_map.items():
        out[canon] = raw[src] if src else None

    out["caller_number"] = out["caller_number"].map(normalize_phone)
    out["callee_number"] = out["callee_number"].map(normalize_phone)
    out["imei"] = out["imei"].map(normalize_imei)
    out["tower_ip"] = out["tower_ip"].map(normalize_ip)
    out["duration_sec"] = pd.to_numeric(out["duration_sec"], errors="coerce").fillna(0)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["source_file"] = os.path.basename(filepath)
    out = out.dropna(subset=["timestamp"]).drop_duplicates()
    return out


def load_chat_logs(filepath: str, verify_integrity: bool = True) -> pd.DataFrame:
    """Load a chat/email export and normalize to CHAT_LOG_COLUMNS."""
    if verify_integrity:
        register_evidence_file(filepath)
    raw = _read_any(filepath)

    col_map = {
        "message_id": _first_present(raw, ["message_id", "id"]),
        "timestamp": _first_present(raw, ["timestamp", "date", "sent_at"]),
        "sender": _first_present(raw, ["sender", "from"]),
        "receiver": _first_present(raw, ["receiver", "to"]),
        "platform": _first_present(raw, ["platform", "app"]),
        "content": _first_present(raw, ["content", "message", "body", "text"]),
    }

    out = pd.DataFrame()
    for canon, src in col_map.items():
        out[canon] = raw[src] if src else None

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["source_file"] = os.path.basename(filepath)
    out = out.dropna(subset=["timestamp"]).drop_duplicates()
    return out


def load_case_evidence(evidence_config: dict) -> dict:
    """
    Convenience loader for a whole case.

    evidence_config example:
        {
            "transactions": ["sample_data/upi_transactions.csv"],
            "call_logs": ["sample_data/cdr_ipdr.csv"],
            "chat_logs": ["sample_data/chat_export.json"],
        }

    Returns a dict of concatenated, normalized DataFrames keyed by type.
    """
    result = {}
    loaders = {
        "transactions": load_transactions,
        "call_logs": load_call_logs,
        "chat_logs": load_chat_logs,
    }
    for kind, paths in evidence_config.items():
        loader = loaders.get(kind)
        if not loader:
            continue
        frames = [loader(p) for p in paths]
        result[kind] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result
