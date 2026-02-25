from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("P%Y%m%d_%H%M%S")
