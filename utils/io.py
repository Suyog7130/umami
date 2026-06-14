
import os
import json
import pickle
import glob
from typing import Any, Dict, Optional
import numpy as np


def ensure_dir(path: str) -> None:
    """ NOTE: All dir paths should end with a '/' to distinguish from file paths. """
    if not path.endswith('/'):
        raise ValueError("Directory path must end with a '/'")
    os.makedirs(path, exist_ok=True)

def ensure_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
def ensure_dirs_and_files(paths: list) -> None:
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if path.endswith('/'):
            ensure_dir(path)
        else:
            ensure_file(path)

def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)

def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(_make_json_serializable(obj), f, indent=2)

def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def save_txt(obj: str, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(obj)

def write_DONE_file(outdir: str, label: str) -> None:
    save_txt("", os.path.join(outdir, f"{label}_DONE"))