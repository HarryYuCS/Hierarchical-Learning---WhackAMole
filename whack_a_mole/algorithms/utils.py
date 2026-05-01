import numpy as np

def flatten_observation(obs) -> np.ndarray:
    """
    Standardizes various observation formats to a numpy array of float32

    Tries to either extract it as the first element of a tuple, as value in dict with key
    "observation", and if can't match these formats will hard return obs as nparray itself.
    
    Args:
        obs : a varying data type containing the observations in some form
    """
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if "observation" in obs:
            return np.asarray(obs["observation"], dtype=np.float32)
        parts = [np.asarray(v, dtype=np.float32).ravel() for _, v in sorted(obs.items())]
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(obs, dtype=np.float32)