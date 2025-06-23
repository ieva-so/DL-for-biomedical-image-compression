
#initialize
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\datasets\EMPIAR-12592\empiar-12592-0000-0900\12592\data-cropped'
com_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\datasets\EMPIAR-12592\empiar-12592-0000-0900\12592\data-cropped'
decom_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\datasets\EMPIAR-12592\empiar-12592-0000-0900\12592\data-processed'

#depenendencies
import numpy as np
import skimage
from skimage import io, filters, color, measure
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from get_psnr import get_psnr
from skimage.metrics import structural_similarity as ssim



# evaluation
def evaluate_images(true_images_dir, com_images_dir, decom_images_dir):
    dataframe = pd.DataFrame(columns=['image_name', 'compression_ratio', 'compression_factor', 'mse','psnr', 'ssim'])
    true_images = sorted(os.listdir(true_images_dir))
    com_images = sorted(os.listdir(com_images_dir))
    decom_images = sorted(os.listdir(decom_images_dir))
    indx = 0
    for true_img_name, com_img_name, decom_img_name in zip(true_images, com_images, decom_images):
        indx = indx+1
        true_img_path = os.path.join(true_images_dir, true_img_name)
        com_img_path = os.path.join(com_images_dir, com_img_name)
        decom_img_path = os.path.join(decom_images_dir, decom_img_name)

        true_img = io.imread(true_img_path)
        decom_img = io.imread(decom_img_path)
        com_img = io.imread(com_img_path)

        #get the compression ratio and factor
        true_im_size = os.path.getsize(true_img_path)
        com_im_size = os.path.getsize(com_img_path)

        compression_ratio = true_im_size / com_im_size
        compression_factor = 1 / compression_ratio 

        psnr = get_psnr(true_img, decom_img)
        mse = np.square(np.subtract(true_img,decom_img)).mean()
        ssim_value = ssim(true_img, decom_img, channel_axis=-1)
        dataframe.loc[indx] = [true_img_name, compression_ratio, compression_factor, mse, psnr, ssim_value]


    return dataframe

df = evaluate_images(true_images_dir, com_images_dir, decom_images_dir)


csv_output_path = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\evaluation_results.csv'
df.to_csv(csv_output_path, index=False)

print(f"Evaluation results saved to: {csv_output_path}")