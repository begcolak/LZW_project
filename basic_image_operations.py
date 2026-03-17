from PIL import Image
import numpy as np

def read_image_from_file(file_path):
    """Opens an image from a given file path using PIL."""
    return Image.open(file_path)

def write_image_to_file(img_obj, file_path):
    """Exports a PIL Image object to a BMP file."""
    img_obj.save(file_path, 'bmp')

def color_to_gray(img_obj):
    """
    Manually converts an RGB image to grayscale using luminosity formula.
    Formula: 0.299*R + 0.587*G + 0.114*B
    """
    data = np.array(img_obj).astype(np.float32)

    if len(data.shape) == 3:
        r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]
        grayscale_map = 0.299 * r + 0.587 * g + 0.114 * b
        return Image.fromarray(grayscale_map.astype(np.uint8))

    return img_obj

def normalize_diff_image(diff_array):
    """Normalizes pixel differences (-255 to 255) to visible range (0 to 255)."""
    f_diff = diff_array.astype(np.float32)
    normalized = (f_diff + 255) / 2
    return normalized.astype(np.uint8)

def PIL_to_np(pil_img):
    """Converts PIL Image to a NumPy array."""
    return np.array(pil_img)

def np_to_PIL(np_arr):
    """Converts a NumPy array back to a PIL Image."""
    return Image.fromarray(np.uint8(np_arr))
