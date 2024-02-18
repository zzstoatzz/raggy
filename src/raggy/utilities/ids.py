import uuid


def generate_prefixed_uuid(prefix: str) -> str:
    """Generate a UUID string with the given prefix."""
    if "_" in prefix:
        raise ValueError("Prefix must not contain underscores.")
    return f"{prefix}_{uuid.uuid4()}"
