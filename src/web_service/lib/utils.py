import pickle
from functools import lru_cache


@lru_cache
def save_pickle(path: str, obj: dict):
    """
    Save an object as a pickled file.

    Args:
        path (str): The path to save the pickle file.
        obj (dict): The object to be pickled and saved.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@lru_cache
def load_pickle(path: str):
    """
    Load a pickled object from a file.

    Args:
        path (str): The path to the pickle file.

    Returns:
        The deserialized object loaded from the file.
    """
    with open(path, "rb") as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
