import os
def rename_files(dirname, prepend):
    files = os.listdir(dirname)
    file_num = len(files)
    for i, f in enumerate(files, start=1):
        new = prepend + f
        file_path = os.path.join(dirname, f)
        dest = os.path.join(dirname, new)
        os.rename(file_path, dest)
        print(f'Moved {file_path} to {dest} {i}/{file_num}')

if __name__ == '__main__':
    pass
