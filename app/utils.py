import numpy as np
from PIL import Image

def preprocess_image(canvas_data):
    """
    Takes the raw drawing canvas input from Gradio, preprocesses it,
    and returns a formatted array ready for the MNIST CNN model.
    """
    if canvas_data is None:
        return None
        
    # 1. Extract the image from the Gradio canvas data dictionary
    # Gradio provides a dictionary containing 'background' and 'layers'
    raw_image = canvas_data["composite"]
    
    # 2. Convert to grayscale ('L' mode)
    # This reduces the image from RGB (3 channels) to 1 channel
    gray_image = raw_image.convert("L")
    
    # 3. Resize to 28x28 pixels
    # Image.Resampling.LANCZOS ensures we keep crisp edges when shrinking
    resized_image = gray_image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 4. Convert to a numpy array
    img_array = np.array(resized_image)
    
    # 5. MNIST check: Ensure it's white digits on a black background.
    # If the canvas center is mostly white, we need to invert the image.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    # 6. Normalize the pixel values to be between 0.0 and 1.0
    img_array = img_array / 255.0
    
    # 7. Reshape to match the model's expected batch input shape: (1, 28, 28, 1)
    # (Batch size of 1, height, width, 1 color channel)
    final_input = img_array.reshape(1, 28, 28, 1)
    
    return final_input