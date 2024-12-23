import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_file(file_path):
    image_file = os.listdir(file_path)
    list_image = []
    for file in image_file:
        image = cv2.imread(os.path.join(file_path, file))
        if image is not None:
            image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            list_image.append(image2)
        else:
            print(f"Warning: Unable to read file {file}")
    return list_image

def resize_image(image, target_size):
    return cv2.resize(image, target_size, interpolation = cv2.INTER_AREA)

def caculate_image(image):
    mean = np.mean(image, axis=(0,1,2))
    std = np.std(image, axis=(0,1,2))
    return mean, std

def normalize_image(image, mean, std):
    return (image - mean) / std

def display_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
# list_image  = read_file("D:\AIO_ALL\Image_retrival\image_retrieval_dataset\images_mr")
# image = list_image[0]

# image = resize_image(image, (64,64))

# mean, std = caculate_image(image)

# image_normalized = normalize_image(image, mean, std)

# display_image(image)