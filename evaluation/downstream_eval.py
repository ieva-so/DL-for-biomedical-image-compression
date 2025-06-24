from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
import stardist.matching as matching
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import io
import os
import pandas as pd
# initialize
compression_model = 'compression_model'
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\datasets\.gitignore\BBBC005_v1_images'
decom_images_dir = true_images_dir + '-decompressed'
stardist_model = '2D_versatile_fluo' #for fluorescence images
dataset_name = 'synthetic fluo'
downstream_eval_csv = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\downstream_eval_results.csv'

#load the stardist model
model = StarDist2D.from_pretrained(stardist_model)

# parse through images in true_images_dir and apply the model to segment them
if os.path.isfile(downstream_eval_csv):
    try:
        dataframe = pd.read_csv(downstream_eval_csv)
    except Exception as e:
        print(f"File exists but failed to read as DataFrame: {e}")
else:
    dataframe = pd.DataFrame(columns=['model', 'dataset', 'image_name', 'dsc','iou'])


for img_name in os.listdir(true_images_dir):
    true_img_path = os.path.join(true_images_dir, img_name)
    decom_img_path = os.path.join(decom_images_dir, img_name)
    true_img = io.imread(true_img_path)
    decom_img = io.imread(decom_img_path)

    true_labels = true_labels, _ = model.predict_instances(normalize(true_img))
    decom_labels = decom_labels, _ = model.predict_instances(normalize(decom_img))
    # Calculate the Dice Similarity Coefficient (DSC)
    result = matching(true_labels, decom_labels, threshold=0.5)
    dsc = result['dsc']
    IoU = result['iou']

    

    dataframe = dataframe.append({
        'model': compression_model,
        'dataset': dataset_name,
        'image_name': img_name,
        'dsc': dsc,
        'iou': IoU
    }, ignore_index=True)
# Save the results to a CSV file
dataframe.to_csv(downstream_eval_csv, index=False)


