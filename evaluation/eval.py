
#initialize
csv_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\evaluation_results.csv'
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\data\dataset\rawimages'
com_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\data\dataset\rawimages-compressed'
decom_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\data\dataset\rawimages-decompressed'
model = 'INN_VAE'
dataset = 'S-BIAD634'
pixel_bits = 8  # 8 for grayscale, 24 for RGB



#dependencies
import numpy as np
import skimage
from skimage import io, filters, color, measure
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from get_psnr import get_psnr
from skimage.metrics import structural_similarity as ssim







if os.path.isfile(csv_dir):
    try:
        dataframe = pd.read_csv(csv_dir)
    except Exception as e:
        print(f"File exists but failed to read as DataFrame: {e}")
else:
    dataframe = pd.DataFrame(columns=['model','dataset','image_name', 'compression_ratio', 'compression_factor','bpp', 'mse','psnr', 'ssim'])

num_rows = len(dataframe)



# evaluation
def evaluate_images(true_images_dir, com_images_dir, decom_images_dir, dataframe):
    true_images = sorted(os.listdir(true_images_dir))
    com_images = sorted(os.listdir(com_images_dir))
    decom_images = sorted(os.listdir(decom_images_dir))
    indx = len(dataframe)
    for true_img_name, com_img_name, decom_img_name in zip(true_images, com_images, decom_images):
        
        true_img_path = os.path.join(true_images_dir, true_img_name)
        com_img_path = os.path.join(com_images_dir, com_img_name)
        decom_img_path = os.path.join(decom_images_dir, decom_img_name)

        true_img = io.imread(true_img_path)
        decom_img = io.imread(decom_img_path)

        #get the compression ratio and factor
        true_im_size = os.path.getsize(true_img_path)
        true_im_dims = true_img.shape
        true_im_pixels = true_im_dims[0] * true_im_dims[1] * true_im_dims[2] if len(true_im_dims) > 2 else true_im_dims[0] * true_im_dims[1]
        com_im_size = os.path.getsize(com_img_path)

        compression_ratio = true_im_size / com_im_size
        compression_factor = 1 / compression_ratio 

        psnr = get_psnr(true_img, decom_img) # Peak Signal-to-Noise Ratio
        mse = np.square(np.subtract(true_img,decom_img)).mean() # Mean Squared Error
        ssim_value = ssim(true_img, decom_img, channel_axis=-1) # Structural Similarity Index
        bpp = com_im_size*pixel_bits / true_im_pixels #Bitrate
        dataframe.loc[indx] = [model, dataset, true_img_name, compression_ratio, compression_factor,bpp, mse, psnr, ssim_value]
        indx = indx+1

    return dataframe

df = evaluate_images(true_images_dir, com_images_dir, decom_images_dir, dataframe)


df.to_csv(csv_dir, index=False)

print(f"Evaluation results saved to: {csv_dir}")