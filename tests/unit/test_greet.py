"""
Unit tests for the greet module.

Tests the get_greeting function to ensure it returns appropriate
greeting messages with and without personalization.
"""

import pytest
from fractal_agent.utils.greet import get_greeting


def test_get_greeting_with_name():
    """Test that get_greeting returns personalized greeting with a name."""
    result = get_greeting("Alice")
    assert result == "Hello, Alice! Welcome!"
    assert "Alice" in result


def test_get_greeting_without_name():
    """Test that get_greeting returns generic greeting without a name."""
    result = get_greeting()
    assert result == "Hello! Welcome!"


def test_get_greeting_with_none():
    """Test that get_greeting handles None explicitly."""
    result = get_greeting(None)
    assert result == "Hello! Welcome!"


def test_get_greeting_with_empty_string():
    """Test that get_greeting with empty string returns generic greeting."""
    result = get_greeting("")
    assert result == "Hello! Welcome!"


def test_get_greeting_with_special_characters():
    """Test that get_greeting handles names with special characters."""
    result = get_greeting("René")
    assert result == "Hello, René! Welcome!"
    assert "René" in result


def test_get_greeting_with_long_name():
    """Test that get_greeting handles longer names correctly."""
    long_name = "Alexander Hamilton"
    result = get_greeting(long_name)
    assert result == f"Hello, {long_name}! Welcome!"
    assert long_name in result


if __name__ == "__main__":
    # Simple example usage for manual verification
    print("Example usage of get_greeting function:")
    print(get_greeting())
    print(get_greeting("World"))
    print(get_greeting("Python Developer"))

    # Run tests if pytest is not being used
    print("\nRunning manual tests:")
    test_get_greeting_with_name()
    print("✓ test_get_greeting_with_name passed")

    test_get_greeting_without_name()
    print("✓ test_get_greeting_without_name passed")

    test_get_greeting_with_none()
    print("✓ test_get_greeting_with_none passed")

    test_get_greeting_with_empty_string()
    print("✓ test_get_greeting_with_empty_string passed")

    test_get_greeting_with_special_characters()
    print("✓ test_get_greeting_with_special_characters passed")

    test_get_greeting_with_long_name()
    print("✓ test_get_greeting_with_long_name passed")

    print("\nAll tests passed!")
