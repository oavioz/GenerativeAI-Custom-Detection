import os

import cv2
import numpy as np

def image_processing(image_path, gamma=1.0, kernel_size=21):
    # Load the image
    original_image = cv2.imread(image_path)

    # Gamma Correction
    gamma_corrected_image = np.power(original_image / 255.0, 1 / gamma) * 255.0
    gamma_corrected_image = gamma_corrected_image.astype(np.uint8)

    # De-blurring using Wiener deconvolution
    psf = np.ones((kernel_size, kernel_size)) / kernel_size**2
    deblurred_image = cv2.filter2D(gamma_corrected_image, -1, psf)

    return deblurred_image

TEST = True

if not TEST:
    input_images_path = "home/ubuntu/projects/LLaVA/playground/cropped_red_images"
    output_dir = "home/ubuntu/projects/LLaVA/playground/fixed_cropped_red_images"

    if not os.path.exists(output_dir):
        # make dir
        os.mkdir(output_dir)

    img_list = os.listdir(input_images_path)


    # Example usage
    input_image_path = "input.jpg"
    output_image = image_processing(input_image_path, gamma=2.2, kernel_size=21)
    cv2.imwrite("output.jpg", output_image)

else:
    input_img = "/home/ubuntu/projects/LLaVA/playground/blured_cropped_red_images/H1rHpfGEW116_19_0_850_1303_0_x-large_0.jpg"
    output_img = "/home/ubuntu/projects/LLaVA/playground/H1rHpfGEW116_19_0_850_1303_0_x-large_0_test.jpg"
    output_image = image_processing(input_img, gamma=2.0, kernel_size=3)
    cv2.imwrite(output_img, output_image)
