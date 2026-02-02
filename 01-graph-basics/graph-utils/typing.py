"""
TypedDist vs Dist
In Python's typing module, `TypedDict` and `Dict` serve different purposes for type hinting dictionaries.
- `Dict`: A generic dictionary type that allows you to specify the types of keys and values. For example, `Dict[str, int]` indicates a dictionary with string keys and integer values. However, it does not enforce any specific structure or required keys.
- `TypedDict`: A more structured way to define dictionaries with specific keys and their corresponding value types. It allows you to create a dictionary type with a fixed set of keys, each associated with a specific type. This is useful for defining data structures with known fields.
"""

from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int



"""
Python’s typing module provides powerful tools for type hinting, making code more readable and maintainable.
"""
from typing import Optional, Any, Annotated, Union, Sequence

#1. Optional:  value can be of a type or None. Used for arguments or variables that may be missing.

def process_optional_value(value: Optional[int]) -> str:
    # Returns info about an int or None
    if value is None:
        return "No value provided"
    return f"Integer value: {value}"

print(process_optional_value(42))      # Integer value: 42
print(process_optional_value(None))    # No value provided


# 2. Any:  value can be any type. Disables type checking for that variable.

def process_any_value(value: Any) -> str:
    return f"Value: {value}"

print(process_any_value(42))           # Value: 42
print(process_any_value("Hello"))      # Value: Hello

# 3. Union:  value can be one of several types (e.g., int or str).

def process_value(value: Union[int, str]) -> str:
    if isinstance(value, int):
        return f"Integer value: {value}"
    elif isinstance(value, str):
        return f"String value: {value}"
    else:
        raise ValueError("Unsupported type")

print(process_value(42))               # Integer value: 42
print(process_value("Hello"))          # String value: Hello


# 4. Sequence:  accepts list, tuple, or any sequence type.

def sum_sequence(seq: Sequence[int]) -> int:
    # Sums any sequence of integers (list, tuple, etc.)
    return sum(seq)

print(sum_sequence([1, 2, 3]))         # 6
print(sum_sequence((4, 5, 6)))         # 15

# 5.Annotated: Add extra info to types; messageType is an annotated string, but annotation is only for type hints and does not enforce length.

MessageType = Annotated[str, "should be of length 5"]
messages: MessageType = "Hello"
print(messages)

