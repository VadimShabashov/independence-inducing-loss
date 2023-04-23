import os
import pathlib


def get_path_to_data_dir():
    return os.path.join(pathlib.Path(__file__).parent.resolve(), 'data')
