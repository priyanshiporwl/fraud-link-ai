"""
utils.py
Shared helper functions: SHA-256 evidence hashing, ID normalization, logging.
"""

import hashlib
import os
import re
import json
from datetime import datetime


def sha256_file(filepath: str) -> str:
    """Compute SHA-256 hash of a file (for forensic integrity verification)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_integrity(filepath: str, expected_hash: str) -> bool:
    """Re-hash a file and compare against a previously stored hash."""
    return sha256_file(filepath) == expected_hash


def load_hash_manifest(manifest_path: str) -> dict:
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def save_hash_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def register_evidence_file(filepath: str, manifest_path: str = "output/evidence_manifest.json") -> str:
    """
    Hash a newly ingested evidence file and record it in the manifest.
    Returns the hash. Call verify_file_integrity() later to detect tampering.
    """
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    manifest = load_hash_manifest(manifest_path)
    file_hash = sha256_file(filepath)
    manifest[os.path.basename(filepath)] = {
        "path": filepath,
        "sha256": file_hash,
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    save_hash_manifest(manifest_path, manifest)
    return file_hash


# ---------- Normalization helpers ----------

def normalize_phone(value) -> str:
    """Strip spaces/dashes/country-code noise from a phone number for matching."""
    if value is None:
        return ""
    digits = re.sub(r"\D", "", str(value))
    # Keep last 10 digits (common Indian mobile number length) for matching
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_upi(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def normalize_account(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s", "", str(value)).upper()


def normalize_ip(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_imei(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\D", "", str(value))
