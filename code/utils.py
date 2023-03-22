import os

def count_files(path):
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])


def get_filenames(path):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]