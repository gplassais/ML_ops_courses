# Use this module to code a `pickle_object` function. This will be useful to pickle the model (and encoder if need be).
import pickle

def save_pickle(path: str, obj:dict):
    """
    Save an object as a pickled file.

    Args:
        path (str): The path to save the pickle file.
        obj (dict): The object to be pickled and saved.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)