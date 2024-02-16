import os
import shutil


def remove_dir(dir_name):
    try:
        shutil.rmtree(dir_name)
    except:
        pass


def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except:
        pass


def create_file(filename):
    try:
        f = open(filename, "w")
        f.close()
    except:
        print("File is not created!")


def write_file(filename: str, content: str, mode="w"):
    try:
        f = open(filename, mode)
        f.write(content)
        f.close()
    except IOError:
        print("An IOError occurred while writing to the file.")
