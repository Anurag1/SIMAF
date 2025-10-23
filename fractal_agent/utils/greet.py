def get_greeting(name: str = None) -> str:
    """
    Return an appropriate greeting message.
    
    Args:
        name: Optional name to personalize the greeting.
        
    Returns:
        A greeting message string.
    """
    if name:
        return f"Hello, {name}! Welcome!"
    return "Hello! Welcome!"