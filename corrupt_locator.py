import os
from PIL import Image

#rootdir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRecognition/train/' #change this to the appropriate rootdir. This program will go through subdirectories.
rootdir = '/mnt/raid0/Projects/Kaggle/GoogleLandmarkRetrieval/test/'
total_corrupt = 0 

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith('.jpg'):
            try:
                img = Image.open(os.path.join(subdir,file)) # open the file as an image
                img.verify() #verify the file is indeed an image
                print(os.path.join(subdir,file))
            except(IOError, SyntaxError) as e:
                print('Corrupt file:',subdir + '/' + file)
                total_corrupt = total_corrupt + 1
                #os.remove(os.path.join(subdir,file)) # Delete the corrupt image
                print('Removed: ', subdir + '/' + file)

print('Total Corrupt: ', total_corrupt)
        
