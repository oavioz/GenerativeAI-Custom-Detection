import os
import cv2
from PIL import Image, ImageFilter, ImageDraw
from ultralytics import YOLO
import logging
import numpy as np


class YoloPersonDetector:
    '''
    Using the popular library "YOLO", we return the bounding boxes
    of people in a given image, which should contain the full body of a person.
    '''
    def __init__(self, conf=0.3):
        self.model = YOLO("yolov8n.pt")
        self.conf = conf
    def predict(self, image: Image.Image):
        image_np = np.array(image)
        results = self.model.predict(source=image_np, conf=self.conf)
        bounding_boxes = []
        for r in results:
            for b in r.boxes:
                if int(b.cls) == 0:
                    xyxy_list = [int(coor) for coor in b.xyxy[0].tolist()]
                    bbbox_dict = {'x': xyxy_list[0],
                                  'y': xyxy_list[1],
                                  'w': xyxy_list[2] - xyxy_list[0],
                                  'h': xyxy_list[3] - xyxy_list[1]
                                }
                    bounding_boxes.append(bbbox_dict)
            logging.info(f"Number of persons detected: {len(bounding_boxes)}")
            return bounding_boxes

def image_processing(original_image, gamma=2.0, kernel_size=3):
    # Convert the PIL image to a NumPy array
    original_image = np.array(original_image)

    # Gamma Correction
    gamma_corrected_image = np.power(original_image / 255.0, 1 / gamma) * 255.0
    gamma_corrected_image = gamma_corrected_image.astype(original_image.dtype)  # Ensure the same data type

    # De-blurring using Wiener deconvolution
    psf = np.ones((kernel_size, kernel_size)) / kernel_size**2
    deblurred_image = cv2.filter2D(gamma_corrected_image, -1, psf)
    deblurred_image_pil = Image.fromarray(deblurred_image)

    return deblurred_image_pil



def crop_with_padding(pil_image, rectangle_coords, padding_percentage=0.):
    """
    Crop a PIL image with padding around a bounding box while preserving the aspect ratio.

    Args:
        pil_image (PIL.Image.Image): The input PIL image.
        rectangle_coords (dict): A dictionary with keys 'x', 'y', 'w', and 'h' defining the bounding box.
        padding_percentage (float): The percentage to increase the bounding box dimensions.

    Returns:
        PIL.Image.Image: The cropped PIL image.
    """
    # Convert the PIL image to a NumPy array
    image = np.array(pil_image)
    
    # Extract bounding box coordinates
    x, y, w, h = rectangle_coords['x'], rectangle_coords['y'], rectangle_coords['w'], rectangle_coords['h']

    # Calculate the increased width and height
    increase_percentage = padding_percentage

    # Calculate the side length for the square crop based on the longer side
    side_length = max(int(w * (1 + 2 * padding_percentage)), int(h * (1 + 2 * padding_percentage)))

    # Calculate the coordinates for cropping
    new_x = max(0, int(x + (w - side_length) / 2))
    new_y = max(0, int(y + (h - side_length) / 2))
    
    new_width = side_length
    new_height = side_length

    # Crop the region from the image
    cropped_image = Image.fromarray(image)
    cropped_image = cropped_image.crop((new_x, new_y, new_x + new_width, new_y + new_height))

    return cropped_image


def process_and_save_images(images_base_dir, save_path, save_cropped=False, blur_images=False, mark_bbox=False):
    detector = YoloPersonDetector()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_list = os.listdir(images_base_dir)
    for jj, image_name in enumerate(img_list[:5]):
        print("#: ", jj)
        print("image_name: ", image_name)
        if "jpg" not in image_name:
            continue
        # Read Image
        img = Image.open(os.path.join(images_base_dir, image_name)).convert("RGB")
        # Preprocess Image - gamma correction and deblurring
        img = image_processing(img, gamma=2.0, kernel_size=3)
        detector.predict(img)
        detections = detector.predict(img)

        for ii, rectangle_coords in enumerate(detections):
            x, y, w, h = rectangle_coords['x'], rectangle_coords['y'], rectangle_coords['w'], rectangle_coords['h']

            # Crop the region of interest
            if save_cropped:
                region = img.crop((x, y, x + w, y + h))
                region.save(os.path.join(save_path, f'{image_name.rsplit(".", 1)[0]}_{ii}.jpg'))
                continue
            else:
                region = img.crop((x, y, x + w, y + h))

            # Create a blurred version of the original image
            blurred_img = img.copy()
            if blur_images:
                blurred_img = blurred_img.filter(ImageFilter.GaussianBlur(radius=20))  # Adjust the radius as needed
                # Paste the cropped region onto the blurred image
                blurred_img.paste(region, (x, y))

            if mark_bbox:
                # Create a drawing context for adding the bounding box
                draw = ImageDraw.Draw(blurred_img)

                # Define the bounding box coordinates
                box_coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

                # Draw a red bounding box
                draw.polygon(box_coords, outline="red", width=2)

                padding_percentage = 0.3
                blurred_img = crop_with_padding(blurred_img, rectangle_coords, padding_percentage)

            # Save the modified image
            output_name = os.path.join(save_path, f'{image_name.rsplit(".", 1)[0]}_{ii}.jpg')
            blurred_img.save(output_name)

            # print(f'x: {x}, y: {y}, w: {w}, h: {h}')


if __name__ == "__main__":
    images_base_dir = "/data/imgs"

    save_path = "/home/ubuntu/Yoni/LLaVA/playground/red_bbox_cropped"
    save_cropped = False
    blur_images = False
    mark_bbox = True
    process_and_save_images(images_base_dir, save_path, save_cropped, blur_images, mark_bbox)