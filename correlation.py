
"""
correlation.py
Step 2: Entity Correlation.

Finds repeated identifiers (phone numbers, UPI IDs, account numbers,
IMEI/IMSI, IP addresses) across evidence sources and builds a unified
list of "entities" plus the edges (relationships) between them.
"""

import pandas as pd
from collections import defaultdict


ENTITY_TYPES = ["account", "upi", "phone", "imei", "ip"]


def build_entity_edges(evidence: dict) -> pd.DataFrame:
    """
    Turn normalized transaction / call-log / chat-log DataFrames into a flat
    edge list: (entity_a, type_a, entity_b, type_b, relation, evidence_ref, timestamp).
    """
    edges = []

    txns = evidence.get("transactions", pd.DataFrame())
    for _, row in txns.iterrows():
        a = row.get("sender_account") or row.get("sender_upi")
        a_type = "account" if row.get("sender_account") else "upi"
        b = row.get("receiver_account") or row.get("receiver_upi")
        b_type = "account" if row.get("receiver_account") else "upi"
        if a and b:
            edges.append({
                "entity_a": a, "type_a": a_type,
                "entity_b": b, "type_b": b_type,
                "relation": "fund_transfer",
                "amount": row.get("amount"),
                "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })
        # link account <-> upi belonging to same party
        if row.get("sender_account") and row.get("sender_upi"):
            edges.append({
                "entity_a": row["sender_account"], "type_a": "account",
                "entity_b": row["sender_upi"], "type_b": "upi",
                "relation": "same_owner",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })
        if row.get("receiver_account") and row.get("receiver_upi"):
            edges.append({
                "entity_a": row["receiver_account"], "type_a": "account",
                "entity_b": row["receiver_upi"], "type_b": "upi",
                "relation": "same_owner",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })

    calls = evidence.get("call_logs", pd.DataFrame())
    for _, row in calls.iterrows():
        caller, callee = row.get("caller_number"), row.get("callee_number")
        if caller and callee:
            edges.append({
                "entity_a": caller, "type_a": "phone",
                "entity_b": callee, "type_b": "phone",
                "relation": "call_contact",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })
        if caller and row.get("imei"):
            edges.append({
                "entity_a": caller, "type_a": "phone",
                "entity_b": row["imei"], "type_b": "imei",
                "relation": "used_device",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })
        if caller and row.get("tower_ip"):
            edges.append({
                "entity_a": caller, "type_a": "phone",
                "entity_b": row["tower_ip"], "type_b": "ip",
                "relation": "network_access",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })

    chats = evidence.get("chat_logs", pd.DataFrame())
    for _, row in chats.iterrows():
        s, r = row.get("sender"), row.get("receiver")
        if s and r:
            edges.append({
                "entity_a": s, "type_a": "unknown",
                "entity_b": r, "type_b": "unknown",
                "relation": "message_contact",
                "amount": None, "timestamp": row.get("timestamp"),
                "evidence_ref": row.get("source_file"),
            })

    return pd.DataFrame(edges)


def compute_entity_features(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the edge list into per-entity features used by the risk model:
    transaction_count, total_amount, linked_accounts, linked_phone_numbers,
    sim_changes (proxy: distinct IMEIs per phone), transaction_velocity.
    """
    if edges.empty:
        return pd.DataFrame(columns=[
            "entity", "entity_type", "transaction_count", "total_amount",
            "linked_accounts", "linked_phone_numbers", "sim_changes",
            "transaction_velocity",
        ])

    stats = defaultdict(lambda: {
        "entity_type": None,
        "transaction_count": 0,
        "total_amount": 0.0,
        "linked_accounts": set(),
        "linked_phone_numbers": set(),
        "linked_imeis": set(),
        "timestamps": [],
    })

    def touch(entity, etype):
        s = stats[entity]
        if s["entity_type"] is None:
            s["entity_type"] = etype
        return s

    for _, e in edges.iterrows():
        sa = touch(e["entity_a"], e["type_a"])
        sb = touch(e["entity_b"], e["type_b"])

        if e["relation"] == "fund_transfer":
            for s in (sa, sb):
                s["transaction_count"] += 1
                s["total_amount"] += float(e["amount"] or 0)
                s["timestamps"].append(e["timestamp"])
            if e["type_a"] == "account":
                sb["linked_accounts"].add(e["entity_a"])
            if e["type_b"] == "account":
                sa["linked_accounts"].add(e["entity_b"])

        if e["relation"] == "call_contact":
            sa["linked_phone_numbers"].add(e["entity_b"])
            sb["linked_phone_numbers"].add(e["entity_a"])

        if e["relation"] == "used_device":
            sa["linked_imeis"].add(e["entity_b"])

        if e["relation"] == "same_owner":
            sa["linked_accounts"].add(e["entity_b"])
            sb["linked_accounts"].add(e["entity_a"])

    rows = []
    for entity, s in stats.items():
        timestamps = sorted([t for t in s["timestamps"] if pd.notna(t)])
        if len(timestamps) >= 2:
            span_hours = max((timestamps[-1] - timestamps[0]).total_seconds() / 3600.0, 0.01)
            velocity = len(timestamps) / span_hours
        else:
            velocity = 0.0

        rows.append({
            "entity": entity,
            "entity_type": s["entity_type"],
            "transaction_count": s["transaction_count"],
            "total_amount": round(s["total_amount"], 2),
            "linked_accounts": len(s["linked_accounts"]),
            "linked_phone_numbers": len(s["linked_phone_numbers"]),
            "sim_changes": max(len(s["linked_imeis"]) - 1, 0),
            "transaction_velocity": round(velocity, 3),
        })

    return pd.DataFrame(rows).sort_values("transaction_count", ascending=False).reset_index(drop=True)
