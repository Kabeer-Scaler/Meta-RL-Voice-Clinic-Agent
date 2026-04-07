"""ID generation utilities."""

import uuid


def generate_episode_id() -> str:
    """
    Generate a unique episode ID.
    
    Returns:
        UUID4 string
        
    Example:
        >>> episode_id = generate_episode_id()
        >>> len(episode_id)
        36
    """
    return str(uuid.uuid4())
