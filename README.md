README for AMND
===
AMND is a system for identification and segmentation of myelinated nerve fibers in a cross-sectional optical-microscopic image using a convolutional neural network. It can extract myelin sheaths from sample images alone. It is optimized for our sample data, however you can customize it in accordance with image sizes, magnifications, and conditions of your samples and what you want to measure.

INSTALL
===
AMND is implemented in [Python] (https://www.python.org) (3.6.0). We recommend you to install [Anaconda] (https://www.continuum.io) since it contains a lot of dependencies for data processing and scientific computing.

Required
---
Two main packages must be installed to run our program. 
* [Chainer] (https://chainer.org/). It is a framework for neural networks. We recommend Chainer 1.13.0 since different versions may cause unexpected errors.
* [OpenCV] (http://opencv.org/). It is a library mainly for computer vision and image processing. We recommend OpenCV 3.2.0 since different versions may cause unexpected errors.
* Several modules are required for using these two packages. Please read the official documentations.

USAGE
===
Download all into your favorite directory. You can put your sample data into "samples" folder and just run "amnd.py". The extensions of sample data must be "bmp", "jpg", "png", or "tiff". Each result output will be saved in "samples" folder with the name of "result_" in the head of each original filename. 

LICENSE
===
The souce code can be modified without notice.  
No claim of suitability, guarantee, or any warranty whatever happens.  
AMND is freely available for non-commercial use. If you are planning on using AMND in a commercial application, please contact us.
