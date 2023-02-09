import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

# img = np.zeros((320, 640, 3), dtype=np.uint8)
img = np.array(Image.open('bunny.jpg'))
img = np.pad(img, ((300, 300), (200, 200), (0, 0)), 'constant')

# # img = np.zeros((320, 640, 3), dtype=np.uint8)
# img = np.array(Image.open('bunny.jpg'))
# # img2 = np.asarray(img.copy())

# img = np.asarray(img)[:, :, 2]
# # img2[:, :, 1] = np.zeros([img2.shape[0], img2.shape[1]])
# # print(img2.shape)
# # img_expanded = np.expand_dims(img, axis=0)
# # print(img_expanded.shape)

# # a = np.arange(-1, 1)
# # a = np.clip(a, -0.5, 0.5)
# # print(a)

plt.imshow(img)
plt.show()

# import skimage
# from skimage import io
# import matplotlib.pyplot as plt
# import scipy
# from scipy import signal
# import numpy as np
# from PIL import Image 

# origImage = Image.open('dog.jpeg').convert('L')
# origImage = np.array(origImage)
# print(origImage.shape)
# filter_kernel = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
# res = scipy.ndimage.convolve(origImage, filter_kernel, mode='constant', cval=0.0)

# plt.imshow(res)
# plt.show()
