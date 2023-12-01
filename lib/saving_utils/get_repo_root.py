import os
from pathlib import Path


def get_repo_root():
    path_var = Path(os.getcwd())
    while path_var.name != 'SS-DDPM':
        path_var = path_var.parent
    return path_var