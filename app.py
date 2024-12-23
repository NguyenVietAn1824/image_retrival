from util import read_file, resize_image, caculate_image, normalize_image, display_image
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(input_image_path, output_image_path, width, height):
    list_image  = read_file(input_image_path)
    list_image_resized = [resize_image(image, (width, height)) for image in list_image]
    list_mean_std = [caculate_image(image) for image in list_image_resized]
    list_image_normalized = [normalize_image(image, mean, std) for image, (mean, std) in zip(list_image_resized, list_mean_std)]
    return list_image_normalized
    for i, image in enumerate(list_image_normalized):
        cv2.imwrite(output_image_path + f"image_{i}.jpg", image)
    
def process_query(query_path):
    query = cv2.imread(query_path)
    query_image_resized = resize_image(query, (64,64))
    mean, std = caculate_image(query_image_resized)
    query_image_normalized = normalize_image(query_image_resized, mean, std)
    return query_image_normalized

def caculate_cosine_similarity(query_image, list_image_normalized):
    similarity = []
    for image in list_image_normalized:
        similarity.append(np.dot(query_image.flatten(), image.flatten()) / (np.linalg.norm(query_image) * np.linalg.norm(image)))
    return similarity

def ranking_image(similarity):
    return np.argsort(similarity)[::-1]

def display_result(input_image_path, ranking_image):
    list_image = read_file(input_image_path)
    for i in ranking_image:
        display_image(list_image[i])

list_image_normalized = process_image("D:\AIO_ALL\Image_retrival\image_retrieval_dataset\images_mr", "D:\AIO_ALL\Image_retrival\image_retrieval_dataset\images_mr_normalized/", 64, 64)
query = process_query("image_retrieval_dataset/images_mr/1.jpg")

similarity = caculate_cosine_similarity(query, list_image_normalized)
ranking = ranking_image(similarity)
plt.imshow(query)
plt.axis("off")
plt.show()
display_result("D:\AIO_ALL\Image_retrival\image_retrieval_dataset\images_mr", ranking)