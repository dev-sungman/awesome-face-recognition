import os

def make_dir(input_path):
    try:
        if not os.path.isdir(input_path):
            os.makedirs(os.path.join(input_path))

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory")
            raise

def remove_dir(input_path):
    try:
        if os.path.isdir(input_path):
            shutil.rmtree(input_path)

    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to delete directory")
            raise
