import os


dirname = 'images/river2'
length = len('river')
for file in os.listdir(dirname):
    new_name = file[:length] + '2-' + file[length:]
    os.rename(os.path.join(dirname, file), os.path.join(dirname, new_name))
