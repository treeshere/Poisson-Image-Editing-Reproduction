Introduction
Poisson image editing is a technique used to seamlessly blend an object or texture from a source image into a target image. This project involves implementing the algorithm in Python and demonstrating its effectiveness with sample images.

Project Overview
Source Code
The primary implementation of the Poisson image editing algorithm is contained in the "poisson_image_editing.py" script. This script performs the necessary computations to blend the source and target images seamlessly. The implementation of the code has referenced Zhou's work.

Images
The project uses the following images:

"source.png": The source image containing the object or texture to be blended.
"target.png": The target image into which the source image will be blended.
"result.png": The resulting image after applying the Poisson image editing algorithm.
Running the Code
To run the code, execute the following command in your terminal:
python poisson_image_editing.py -s <source image> -t <target image> [-m <mask>]

Ensure that the source and target images are placed in the same directory with "poisson_image_editing.py".

If the mask image is not specified, a window will pop up for you to draw the mask on the source image:
The green region will be used as the mask. Press "s" to save the result, press "r" to reset.

After the mask is defined, a window will pop up for you to adjust the mask position on the target image:
The mask corresponds to the region of the source image that will be copied. You can move the mask to place the copied part into the desired location in the target image. Press "s" to save the result, press "r" to reset.

Then the Poisson image editing process will start. The blended image will be named as "target_result.png", in the same directory as the source image.
