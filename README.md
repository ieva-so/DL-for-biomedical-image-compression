# compression (we are not compressing dl)
This is our repo for the DL biomedical image analysis papers
TIMELINE
![image](https://github.com/user-attachments/assets/a8eefb84-155e-4cc1-833c-637e7be301ff)
TASKS
![image](https://github.com/user-attachments/assets/209a6474-d6d7-4eed-abd3-040a617e0886)

IMPORTANT: have to allow to switch between architectures, which include or exclude colour component!

## Training:

Loss functions https://arxiv.org/pdf/1511.08861
for biomedical images (structure is important) - compression rate + MS-SSIM (

Grayscale: EMPIAR-12592 (Scanning Electron Microscopy)
RGB: Open TG-GATEs (Histopathology)

Benchmarking stage 1: Compression benchmarking dataset  
  - https://imagecompression.info/test_images/  
  - Grayscale 16 bit - get standard benchmarking metrics out  
  - RGB 16 bit - get standard benchmarking metrics out  
Benchmarking stage 2: Performance on grayscale images - Electron microscopy dataset (https://www.ebi.ac.uk/empiar/EMPIAR-12592/) PLEASE ADD THIS DATASET, MY INTERNET IS TOO SLOW!!!  
  - General performance (compression benchmark metrics) on EMPIAR-12592 dataset  
  - Segmentation performance (downstream task) on EMPIAR-12592 dataset  
  - Contour extraction performance on EMPIAR-12592 dataset  
Benchmarking stage 3: Performance on RGB images - Histopathology dataset  
  - General compression performance metrics on the Open TG-GATEs dataset  
  - Performance of nuclei segmentation before and after segmentation (STARDIST)  
Benchmarking stage 4: unseen image types  
  - Grayscale: MRI dataset (select one from bioimage archive)  
  - RGB: Brightfield microscopy (select one from bioimage archive)  


## Evaluation Metrics

1) Compression time
2) Compression Ratio (size of compressed image)/(size of original image) via sys.getsizeof(image), inverse - compression factor
3) Information loss: Mean Squared
   ![image](https://github.com/user-attachments/assets/1b369f23-e3cd-4b22-b177-89f54bcb0a3d)
4) peak signal to noise ratio pSNR = 20log20(Imax/sqrt(MSE)) - higher value indicates higher image quality
5) Structural integrity and degradation - Structural similarity index measure (SSIM), implimented from scikit-image, import compare_ssim
   ![image](https://github.com/user-attachments/assets/10c7301b-9a05-4367-932b-601166c6f15c)
6) Segmentation and contour similarity - DICE score ![image](https://github.com/user-attachments/assets/9187cfb4-074c-4ae7-8992-61f98d9826eb), SEGMENT WITH STARDIST (https://arxiv.org/abs/2203.02284)

