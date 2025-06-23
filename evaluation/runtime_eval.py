# initialize
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\datasets\EMPIAR-12592\empiar-12592-0000-0900\12592\data-cropped'
com_images_dir = true_images_dir + '-compressed'
decom_images_dir = com_images_dir + '-decompressed'
model = somemodel.py
weights = 'path to weights'
dataset_name = 'EMPIAR-12592'
runtime_eval_csv = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\runtime_eval_results.csv'

#parse through images in true_images_dir and apply the model to compress and decompress them
import os
import time
from skimage import io, filters, color, measure
from model import encode_image, decode_image  # Assuming these functions are defined in model.py
import pandas as pd

dataframe = pd.DataFrame(columns=['model', 'dataset', 'image_name', 'image_size','time_encode', 'time_decode'])


for img_name in os.listdir(true_images_dir):
    true_img_path = os.path.join(true_images_dir, img_name)
    com_img_path = os.path.join(com_images_dir, img_name)
    decom_img_path = os.path.join(decom_images_dir, img_name)


    #get size of true image
    true_im_size = os.path.getsize(true_img_path)


    # Compress the image
    #start timer
    
    start_time = time.time()    
    compressed_img = encode_image(true_img, model, weights)
    io.imsave(com_img_path, compressed_img)
    end_time = time.time()
    time_encode = end_time - start_time

    # Decompress the image
    start_time = time.time()
    decompressed_img = decode_image(compressed_img, model, weights)
    io.imsave(decom_img_path, decompressed_img)
    end_time = time.time()
    time_decode = end_time - start_time

    dataframe = dataframe.append({
        'model': model,
        'dataset': dataset_name,
        'image_name': img_name,
        'image_size': true_im_size,
        'time_encode': time_encode,
        'time_decode': time_decode
    }, ignore_index=True)
# Save the results to a CSV file
dataframe.to_csv(runtime_eval_csv, index=False)
    
