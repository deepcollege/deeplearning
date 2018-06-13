import os


def file_exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True


def try_create_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
