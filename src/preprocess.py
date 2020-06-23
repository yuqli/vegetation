from skimage import io
import matplotlib.pyplot as plt

data_path = "../S2A_3Band_Cropped.tif"

# read the image stack
img = io.imread(data_path)
# show the image

with open('tmp.txt', 'a') as f:
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                f.write(str(img[i][j][k]) + '\n')