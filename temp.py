import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PIL_Image  # Ensuring Pillow is correctly referenced for image processing

app = Flask(__name__)
model = load_model('model.h5')  # Ensure the model file path is correct

@app.route("/") 
def about():
    return render_template("about.html")

@app.route("/about") 
def home():
    return render_template("about.html")

@app.route("/info") 
def information():
    return render_template("info.html")

@app.route("/upload") 
def test():
    return render_template("index6.html")

@app.route("/predict", methods=["GET", "POST"]) 
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not f:
            return "No file uploaded", 400  # Handling cases where no file is provided

        basepath = os.path.dirname(__file__)  # Corrected to use __file__ for accurate directory referencing
        upload_path = os.path.join(basepath, "uploads")
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)  # Ensuring the upload directory exists
        filepath = os.path.join(upload_path, f.filename)
        f.save(filepath)
        
        img = image.load_img(filepath, target_size=(64, 64))  # Loading and resizing the image
        x = np.asarray(img)
        x = np.expand_dims(x, axis=0)  # Adding batch dimension
        pred = model.predict(x)
        print("Shape of the input to model:", x.shape)
        print("Prediction result:", pred)
        
        # Assuming binary classification with sigmoid activation
        result = "Uninfected" if pred[0][0] > 0.5 else "Infected"
        return result
    else:
        # Provide a fallback for GET requests or unsuccessful POST
        return render_template("index6.html")  

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)  # Using correct syntax for execution guard
