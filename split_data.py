import os
import random
import shutil


def move_files_to_directory(source, dest, fraction):
    files = os.listdir(source)
    no_of_files = int(len(files) * fraction)
    os.makedirs(dest)
    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(source, file_name), dest)

subset_names = ['Mixture', 'NoGas', 'Perfume', 'Smoke']
for name in subset_names:
    source = 'dataset/images/' + name
    dest = 'dataset/train/images/' + name
    move_files_to_directory(source, dest, 0.8) 
    dest = 'dataset/test/images/' + name
    move_files_to_directory(source, dest, 0.5)
    dest = 'dataset/validation/images/' + name
    move_files_to_directory(source, dest, 1)
