# initialize
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\data\dataset\rawimages'
com_images_dir = true_images_dir + '-compressed'
decom_images_dir = true_images_dir + '-decompressed'
model = 'model_name'
weights = 'path to weights'
dataset_name = 'S-BIAD634'
runtime_eval_csv = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\runtime_eval_results.csv'
model_path = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\INN_VAE\evaluation'




#parse through images in true_images_dir and apply the model to compress and decompress them
import os
import time
from skimage import io, filters, color, measure
import sys
if model_path not in sys.path:
    sys.path.append(model_path)
from decompress_file import decompress_file  # Assuming decompress_file.py contains the decompress function
from compression_metrics import 
import pandas as pd

dataframe = pd.DataFrame(columns=['model', 'dataset', 'image_name', 'image_size','time_encode', 'time_decode', 'compression_ratio'])


for img_name in os.listdir(true_images_dir):
    true_img_path = os.path.join(true_images_dir, img_name)
    com_img_path = os.path.join(com_images_dir, img_name)
    decom_img_path = os.path.join(decom_images_dir, img_name)


    #get size of true image
    true_im_size = os.path.getsize(true_img_path)
    true_img = io.imread(true_img_path)


    # Compress the image
    #start timer

    start_time = time.time()
    compressed_img = compress(INPUTS)
    io.imsave(com_img_path, compressed_img)
    end_time = time.time()
    time_encode = end_time - start_time
    com_im_size = os.path.getsize(com_img_path)
    compression_ratio = true_im_size / com_im_size

    # Decompress the image
    start_time = time.time()
    decompressed_img = decompress_file(INPUTS)
    io.imsave(decom_img_path, decompressed_img)
    end_time = time.time()
    time_decode = end_time - start_time

    dataframe = dataframe.append({
        'model': model,
        'dataset': dataset_name,
        'image_name': img_name,
        'image_size': true_im_size,
        'time_encode': time_encode,
        'time_decode': time_decode,
        'compression_ratio': compression_ratio
    }, ignore_index=True)
# Save the results to a CSV file
dataframe.to_csv(runtime_eval_csv, index=False)
    
