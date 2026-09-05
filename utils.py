import numpy as np
from PIL import Image

def preprocess_image(canvas_data):
    """
    Takes the raw drawing canvas input from Gradio, preprocesses it,
    and returns a formatted array ready for the MNIST CNN model.
    """
    if canvas_data is None:
        return None
        
    # Extract the image from the Gradio canvas data dictionary
    raw_image = canvas_data["composite"]
    
    # Convert to grayscale ('L' mode)
    gray_image = raw_image.convert("L")
    
    # Resize to 28x28 pixels
    resized_image = gray_image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to a numpy array
    img_array = np.array(resized_image)
    
    # MNIST check: Ensure it's white digits on a black background.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    # Normalize the pixel values to be between 0.0 and 1.0
    img_array = img_array / 255.0
    
    # Reshape to match the model's expected batch input shape: (1, 28, 28, 1)
    final_input = img_array.reshape(1, 28, 28, 1)
    
    return final_input
