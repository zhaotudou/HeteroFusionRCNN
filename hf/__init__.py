import os


def root_dir():
    return os.path.dirname(os.path.realpath(__file__))


def top_dir():
    hf_root_dir = root_dir()
    return os.path.split(hf_root_dir)[0]
