
import os
import json
import pickle
import glob
from typing import Any, Dict, Optional
import numpy as np


import logging
logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """ NOTE: All dir paths should end with a '/' to distinguish from file paths. """
    if not path.endswith('/'):
        if not os.path.isdir(path) or not os.path.exists(path):
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
    if label=='':
        filename = os.path.join(outdir, "DONE")
    else:
        filename = os.path.join(outdir, f"{label}_DONE")
    save_txt("", filename)


def check_DONE_file_exists(outdir: str, 
                           label: str = '',
                           injection_index: int = None,
                           check_all_subdirs: bool = False) -> bool:
    logger.info(f"Checking for DONE file in {outdir} with label '{label}' and check_all_subdirs={check_all_subdirs}")
    logger.debug(f"outdir partitioned: {outdir.partition('results/')}")
    if check_all_subdirs:
        if outdir.partition('results/')[1] == 'results/':
            outdir = os.path.join(outdir.partition('results/')[0], 'results/*/')
    logger.info(f"Final outdir for checking: {outdir}")
    if injection_index is not None:
        outdir = os.path.join(outdir, f'*{label}*inj_{injection_index}_*/')
    done_file_path = os.path.join(outdir, f"*DONE*")
    logger.debug(f"Looking for DONE files matching: {done_file_path}")
    if check_all_subdirs:
        done_files = glob.glob(done_file_path, recursive=True)
        logger.debug(f"Found DONE files: {done_files}")
        return len(done_files) > 0
    else:
        filename = os.path.join(outdir, "*DONE*")
        logger.debug(f"Checking for file: {filename}")
        return os.path.isfile(filename)