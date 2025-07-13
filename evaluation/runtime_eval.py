# initialize
true_images_dir = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\data\dataset\rawimages'
com_images_dir = true_images_dir + '-compressed'
decom_images_dir = true_images_dir + '-decompressed'
model = 'INN_VAE'
dataset_name = 'S-BIAD634'
runtime_eval_csv = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\evaluation\runtime_eval_results.csv'
model_path = r'C:\Users\ievas\Desktop\UNI\00 SEMESTERS\SoSe25\Project Seminar Biomedical Image Analysis\DL_compression\INN_VAE\evaluation'


#parse through images in true_images_dir and apply the model to compress and decompress them
import time
from skimage import io, filters, color, measure
import imagecodecs
from compress_decompress_modularised import compress, decompress, get_sample  # Assuming decompress_file.py contains the decompress function
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
# Add path to INN_VAE folder so training/ and models/ can be found
current_dir = os.path.dirname(__file__)
inn_vae_path = os.path.abspath(os.path.join(current_dir, '..', 'INN_VAE'))
sys.path.append(inn_vae_path)
from models.inn import build_inn
from training.train_conv_vae import ConvVAE

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")


print("Loading INN and ConvVAE models...")
inn_model = build_inn(channels=3).to(DEVICE)
inn_model.load_state_dict(torch.load("C:\\Users\\ievas\\Desktop\\UNI\\00 SEMESTERS\\SoSe25\\Project Seminar Biomedical Image Analysis\\data\\checkpoints\\inn_model.pth", map_location=DEVICE))
inn_model.eval()

conv_vae = ConvVAE(in_channels=3, latent_dim=2048).to(DEVICE)
conv_vae.load_state_dict(torch.load("C:\\Users\\ievas\\Desktop\\UNI\\00 SEMESTERS\\SoSe25\\Project Seminar Biomedical Image Analysis\\data\\checkpoints\\conv_vae_model_low_compression.pth", map_location=DEVICE))
conv_vae.eval()


os.makedirs(com_images_dir, exist_ok=True)
os.makedirs(decom_images_dir, exist_ok=True)
dataframe = pd.DataFrame(columns=['model', 'dataset', 'image_name', 'image_size','time_encode', 'time_decode', 'compression_ratio'])


for img_name in os.listdir(true_images_dir):
    true_img_path = os.path.join(true_images_dir, img_name)
    com_img_path = os.path.join(com_images_dir, img_name)
    decom_img_path = os.path.join(decom_images_dir, img_name)


    #get size of true image

    # === Load and transform the image ===
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
    ])

    img = Image.open(true_img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dim
    img_name = os.path.basename(true_img_path)
    true_im_size = os.path.getsize(true_img_path)


    true_im_size = os.path.getsize(true_img_path)
    true_img = io.imread(true_img_path)


    # Compress the image
    #start timer

    start_time = time.time()
    compressed_img, z_latent = compress(img_tensor, inn_model, conv_vae)

    torch.save(compressed_img.detach().cpu(), com_img_path.replace('.tif', '.pt'))
    end_time = time.time()
    time_encode = end_time - start_time
    com_im_size = os.path.getsize(com_img_path.replace('.tif', '.pt'))
    compression_ratio = true_im_size / com_im_size

    # Decompress the image
    start_time = time.time()
    decompressed_img = decompress(compressed_img, conv_vae, inn_model)
    decompressed_img = decompressed_img.detach().squeeze(0).cpu().numpy().transpose(1, 2, 0)
    io.imsave(decom_img_path, decompressed_img)
    end_time = time.time()
    time_decode = end_time - start_time

    dataframe.loc[len(dataframe)] = {
    'model': model,
    'dataset': dataset_name,
    'image_name': img_name,
    'image_size': true_im_size,
    'time_encode': time_encode,
    'time_decode': time_decode,
    'compression_ratio': compression_ratio
    }

# Save the results to a CSV file
dataframe.to_csv(runtime_eval_csv, index=False)
print('done')   
