
import os


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