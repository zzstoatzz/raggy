import uuid


def generate_prefixed_uuid(prefix: str) -> str:
    """Generate a UUID string with the given prefix.

    Args:
        prefix: The prefix to use for the UUID

    Returns:
        A UUID string with the given prefix

    Raises:
        ValueError: If the prefix contains an underscore

    Example:
        Generate a UUID with the prefix "my_prefix"
        ```python
        from raggy.utilities.ids import generate_prefixed_uuid

        uuid = generate_prefixed_uuid("schleeb") # 'schleeb_6be20040-b7e0-4990-b271-394221584a59'
        ```

    """
    if "_" in prefix:
        raise ValueError("Prefix must not contain underscores.")
    return f"{prefix}_{uuid.uuid4()}"
