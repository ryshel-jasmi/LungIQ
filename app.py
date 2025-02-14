from pymongo import MongoClient
import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["LungIQ"] 
contact_collection = db["contacts"] 

# Load your pre-trained model with compile=False to suppress compiled metrics warning
model = load_model('models/CNN_Covid19_Xray_Version.h5', compile=False)

# Load the LabelEncoder using the current version of scikit-learn
with open("models/Label_encoder.pkl", 'rb') as f:
    le = pickle.load(f)

# Path to store uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_image(image_path):
    # Read and process the image for model prediction
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Convert to RGB, resize, normalize, and reshape for model input
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (150, 150))
    image_normalized = image_resized / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    # Make prediction
    predictions = model.predict(image_input)
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    predicted_label = le.inverse_transform([predicted_index])[0]
    
    return predicted_label, confidence_score

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success_message = None
    if request.method == 'POST':
        # Retrieve form data
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject', 'No Subject')
        message = request.form.get('message')

        # Insert data into MongoDB
        contact_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "message": message,
        }
        contact_collection.insert_one(contact_data)  # Fix: Corrected the collection name

        # Set success message
        success_message = "Thank you for contacting us! We'll get back to you soon."

    return render_template('contact.html', success_message=success_message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process image and get prediction
        predicted_label, confidence_score = process_image(file_path)
        
        return render_template('result.html',
                               image_path=file_path,
                               filename=filename,
                               predicted_label=predicted_label,
                               confidence_score=confidence_score)

if __name__ == '__main__':
    app.run(debug=True)
