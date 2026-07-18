"""Canonical surveillance schema and validation."""
REQUIRED=("date","disease","region_id","signal_type","value","source")
def validate_columns(frame):
    missing=[c for c in REQUIRED if c not in frame.columns]
    if missing: raise ValueError(f"missing surveillance columns: {missing}")
    return True
