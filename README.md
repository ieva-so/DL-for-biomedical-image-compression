# compression (we are not compressing dl)
This is our repo for the DL biomedical image analysis papers
TIMELINE
![image](https://github.com/user-attachments/assets/a8eefb84-155e-4cc1-833c-637e7be301ff)
TASKS
![image](https://github.com/user-attachments/assets/209a6474-d6d7-4eed-abd3-040a617e0886)

IMPORTANT: have to allow to switch between architectures, which include or exclude colour component!

Training:
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
  - Performance of nuclei segmentation before and after segmentation (SAM for histopathology - https://arxiv.org/abs/2502.00408)  
Benchmarking stage 4: unseen image types  
  - Grayscale: MRI dataset (select one from bioimage archive)  
  - RGB: Brightfield microscopy (select one from bioimage archive)  

