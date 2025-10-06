from typing import Dict, Any, List, Tuple
import functools

def set_nested_attribute(obj: Any, path: Tuple[str, ...], value: Any):
    """
    Sets a nested attribute or dictionary key on an object programmatically.

    Args:
        obj (Any): The root object to modify.
        path (Tuple[str, ...]): A tuple representing the path to the attribute.
                                Each element is an attribute name or a dictionary key.
        value (Any): The new value to set.
    """
    # Navigate to the parent of the target attribute/key
    parent_path = path[:-1]
    final_key = path[-1]

    try:
        # Use functools.reduce to traverse the path. For each step, it either
        # gets an attribute from an object or an item from a dictionary.
        parent = functools.reduce(
            lambda current_obj, key: getattr(current_obj, key) if hasattr(current_obj, key) else current_obj[key],
            parent_path,
            obj
        )

        # Set the final value on the retrieved parent object/dictionary
        if isinstance(parent, dict):
            parent[final_key] = value
        else:
            setattr(parent, final_key, value)
    except (AttributeError, KeyError) as e:
        print(f"Error: Could not find path '{'.'.join(path)}'. Invalid key or attribute: {e}")