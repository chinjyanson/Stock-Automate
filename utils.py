def safe_to_dict(obj):
    """
    Safely convert an object to a dictionary, handling None values.
    """
    return obj.to_dict() if obj is not None else {}