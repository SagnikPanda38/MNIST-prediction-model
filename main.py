import os
import tensorflow as tf
import gradio as gr
import numpy as np
from utils import preprocess_image
import h5py, json

# 1. Path setup to securely load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_cnn_model.h5")
path = MODEL_PATH  # your .h5 file

with h5py.File(path, 'r+') as f:
    model_config = f.attrs['model_config']
    if isinstance(model_config, bytes):
        model_config = model_config.decode('utf-8')
    config = json.loads(model_config)

    def strip_quant(layer_cfg):
        if 'config' in layer_cfg and 'quantization_config' in layer_cfg['config']:
            del layer_cfg['config']['quantization_config']
        # recurse into nested Sequential/Functional configs if present
        if 'config' in layer_cfg and 'layers' in layer_cfg['config']:
            for sub in layer_cfg['config']['layers']:
                strip_quant(sub)

    strip_quant(config)  # if top-level is itself a Sequential wrapper, adjust accordingly
    for layer in config.get('config', {}).get('layers', []):
        strip_quant(layer)

    f.attrs['model_config'] = json.dumps(config)


# Load the pre-trained neural network
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find model at {MODEL_PATH}. Did you move 'mnist_cnn_model.h5' there?")

model = tf.keras.models.load_model(MODEL_PATH)

# 2. Define the prediction function
def predict_digit(canvas):
    if canvas is None:
        return "Please draw a number first!"
        
    # Process the canvas input using our utility function
    processed_input = preprocess_image(canvas)
    
    # Get probabilities for all 10 digits from the model
    predictions = model.predict(processed_input)[0]
    
    # Map each digit (0-9) to its corresponding confidence score
    # Gradio's Label component beautifully displays this as a bar chart
    return {str(i): float(predictions[i]) for i in range(10)}

# 3. Build the Interactive Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✏️ Digital Slate: MNIST Digit Classifier")
    gr.Markdown("Draw a single digit (0-9) clearly in the center of the slate below, then watch the model predict it!")
    
    with gr.Row():
        # Left side: The slate component where the user draws
        with gr.Column():
            canvas_input = gr.Sketchpad(
                label="Draw Here",
                type="pil",
                layers=False,       # Single drawing layer
                brush = gr.Brush(colors=["#000000"], default_size=18, color_mode="fixed") # Black ink, thick brush stroke
            )
            clear_btn = gr.Button("Clear Slate", variant="secondary")
            submit_btn = gr.Button("Predict Digit", variant="primary")
            
        # Right side: The prediction output display
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Top Predictions")
            
    # Connect UI elements to functionality
    submit_btn.click(fn=predict_digit, inputs=canvas_input, outputs=output_label)
    clear_btn.click(fn=lambda: None, outputs=canvas_input) # Resets the canvas

# 4. Launch the web server
if __name__ == "__main__":
    # Setting share=True gives you a public link you can text to your friends!
    demo.launch(share=False)