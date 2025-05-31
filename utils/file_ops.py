def write_file(filename: str, content: str, mode="w"):
    try:
        f = open(filename, mode)
        f.write(content)
        f.close()
    except IOError:
        print("An IOError occurred while writing to the file.")
