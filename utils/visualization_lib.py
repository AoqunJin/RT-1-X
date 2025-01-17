import os
import json
import jax
import numpy as np
from PIL import Image


def add_white_border(image, border_size):
    """
    Add a white border around a single image.
    
    :param image: np.ndarray, shape [h, w, c]
    :param border_size: int, width of the border
    :return: np.ndarray, shape [h+2*border_size, w+2*border_size, c]
    """
    h, w, c = image.shape
    bordered_image = np.ones((h + 2 * border_size, w + 2 * border_size, c), dtype=image.dtype) * 255
    bordered_image[border_size:-border_size, border_size:-border_size, :] = image
    return bordered_image


def save_images_vertically_with_border(data, border_size, output_path):
    """
    Save images with shape [y, x, h, w, c] to a single image file, 
    stacked vertically by y, with a white border around each image.
    
    :param data: np.ndarray, shape [y, x, h, w, c]
    :param border_size: int, width of the white border
    :param output_path: str, file path to save the resulting image
    """
    # Ensure data is in uint8 format for saving as an image
    data = (data * 255).astype(np.uint8) if data.dtype != np.uint8 else data

    # Add white borders to each image
    y, x, h, w, c = data.shape
    bordered_data = np.zeros((y, x, h + 2 * border_size, w + 2 * border_size, c), dtype=data.dtype)
    
    for yi in range(y):
        for xi in range(x):
            bordered_data[yi, xi] = add_white_border(data[yi, xi], border_size)
    
    # Reshape dimensions to [y, (h+2*border_size)*x, (w+2*border_size), c]
    h_with_border, w_with_border = h + 2 * border_size, w + 2 * border_size
    reshaped_images = bordered_data.transpose(0, 2, 1, 3, 4).reshape(y * h_with_border, x * w_with_border, c)
    
    # Convert to PIL Image and save
    image = Image.fromarray(reshaped_images)
    image.save(output_path)
    print(f"Image with borders saved to {output_path}")
    
    
def save_images_vertically(data, output_path):
    """
    Save images with shape [y, x, h, w, c] to a single image file, stacked vertically by y.

    :param data: np.ndarray, shape [y, x, h, w, c]
    :param output_path: str, file path to save the resulting image
    """
    # Ensure data is in uint8 format for saving as an image
    data = (data * 255).astype(np.uint8) if data.dtype != np.uint8 else data

    # Reshape dimensions to [y, x*h, w, c]
    y, x, h, w, c = data.shape
    reshaped_images = data.transpose(0, 2, 1, 3, 4).reshape(y * h, x * w, c)

    # Convert to PIL Image and save
    image = Image.fromarray(reshaped_images)
    image.save(output_path)
    print(f"Image saved to {output_path}")


def __save_batch(batch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {}

    for key, value in batch.items():
        if isinstance(value, (jax.Array, np.ndarray)) and len(value.shape) >= 5:
            # Assume image data, save image separately
            # Image data is expected to have shape [y, x, h, w, c]
            image_filename = os.path.join(output_dir, "sample_batch.png")
            save_images_vertically_with_border(value, border_size=2, output_path=image_filename)
            metadata[key] = str(image_filename)
            
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            metadata[key] = __save_batch(value, output_dir)

        else:
            # Store non-image data as-is
            metadata[key] = value.tolist() if isinstance(value, np.ndarray) else value

    return metadata

        
def save_batch(batch, output_dir):
    """
    Save images and other information from a batch into a dictionary.

    Parameters:
        batch (dict): A dictionary containing images and other information.

    Returns:
        dict: A dictionary with processed data.
    """    
    d_data = __save_batch(batch, output_dir)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(d_data, f, indent=4)
        