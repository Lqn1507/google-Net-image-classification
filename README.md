# google-Net-image-classification
Build
Export a new environment variable that points to your local installation of CUDA's CUTLASS v1.3.2 library:
$ export CUTLASS_INSTALL_DIR=path/to/cutlass

or 
use google colab:

- access google drive:
$ from google.colab import drive
$ drive.mount('/content/drive')

- download cuda 9.1:
  
!wget --no-clobber https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64

!wget --no-clobber https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/1/cuda-repo-ubuntu1704-9-1-local-cublas-performance-update-1_1.0-1_amd64

!wget --no-clobber https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/2/cuda-repo-ubuntu1704-9-1-local-compiler-update-1_1.0-1_amd64

!wget --no-clobber https://developer.nvidia.com/compute/cuda/9.1/Prod/patches/3/cuda-repo-ubuntu1704-9-1-local-cublas-performance-update-3_1.0-1_amd64

- install cuda:
!sudo dpkg -i cuda-repo-ubuntu1704-9-1-local_9.1.85-1_amd64

!sudo dpkg -i cuda-repo-ubuntu1704-9-1-local-compiler-update-1_1.0-1_amd64

!sudo dpkg -i cuda-repo-ubuntu1704-9-1-local-cublas-performance-update-1_1.0-1_amd64

!sudo dpkg -i cuda-repo-ubuntu1704-9-1-local-cublas-performance-update-3_1.0-1_amd64

- move to your file location:

- import imagenet_stubs
 !pip3 install git+https://github.com/nottombrown/imagenet_stubs

Run
!python3 ./run.py -googlenet

dataset have 1000 images but the code only run 200 images. divide the folder into small folder before run it
