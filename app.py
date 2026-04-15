# from transformers import pipeline

# model = pipeline("image-classification", model="DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer", use_fast=True)
# response = model("https://storage.googleapis.com/kagglesdsdata/datasets/601280/1079953/lung_colon_image_set/lung_image_sets/lung_aca/lungaca102.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251024%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251024T151333Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8656da938e88249e9be13cb42d19f15394ef8d2e07d9063eaf25fb29868ae2ed2183156997bca5d6c58110c35c12152f577fcd20198b2998acb229c9596649eb79ad135871762a8bba78d9bfc8748bb46d2a440d6f8bbd9ef08b3ff17c4fa2f31e0eb79eb166ac547c2a8da2eb7a5934e0ea3662e19249b7c14d3c1a2e9929150fe450dc01d09d5d5761dfb05e2a4430d67fa581935fbc73233adc26fc388b819ec4a72b468d62f68f83ca293532d625ac3d523ada86573e44f90aa26292ac7272318891e2e8ab6bcedbe71a990101ca9eb84d233871878975dc2dac8cb57eeece92679d98125ef51584fbfed979cfa95e821bd04ad01d3a06c087ea82e1ff21")

# print(response)

from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import pipeline
from PIL import Image
import os
import json
from urllib.parse import quote
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
from werkzeug.utils import secure_filename
import torch
from dotenv import load_dotenv

load_dotenv()
WAQI_API_TOKEN = os.getenv('WAQI_API_KEY')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
print("Loading model...")
model = pipeline(
    "image-classification", 
    model="DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer",
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_label(label):
    """Convert label codes to readable format"""
    label_map = {
        'lung_aca': 'Lung Adenocarcinoma',
        'lung_scc': 'Lung Squamous Cell Carcinoma',
        'lung_n': 'Lung Normal',
        'colon_aca': 'Colon Adenocarcinoma',
        'colon_n': 'Colon Normal'
    }
    return label_map.get(label, label)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cities.js')
def cities_js():
    return send_from_directory('templates', 'cities.js')

@app.route('/style.css')
def style_css():
    return send_from_directory('templates', 'style.css')

@app.route('/api/air-quality', methods=['GET'])
def get_air_quality():
    city = request.args.get('city', '').strip()
    if not city:
        return jsonify({'error': 'City is required'}), 400

    token = os.environ.get('WAQI_API_TOKEN', 'demo')
    waqi_url = f"https://api.waqi.info/feed/{quote(city)}/?token={WAQI_API_TOKEN}"

    try:
        with urlopen(waqi_url, timeout=10) as response:
            payload = json.loads(response.read().decode('utf-8'))

        if payload.get('status') != 'ok':
            return jsonify({
                'error': payload.get('data', 'Unable to fetch air quality data from WAQI')
            }), 400

        return jsonify({'success': True, 'data': payload.get('data', {})})

    except (HTTPError, URLError):
        return jsonify({'error': 'Failed to connect to WAQI API'}), 502
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    cancer_type = request.form.get('cancer_type', 'both')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process the image
            image = Image.open(filepath)
            
            # Get predictions
            results = model(image)
            
            # Filter results based on cancer type selection
            if cancer_type == 'lung':
                results = [r for r in results if r['label'].startswith('lung_')]
            elif cancer_type == 'colon':
                results = [r for r in results if r['label'].startswith('colon_')]
            
            # Format results
            formatted_results = [
                {
                    'label': format_label(r['label']),
                    'raw_label': r['label'],
                    'score': round(r['score'] * 100, 2)
                }
                for r in results
            ]
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'results': formatted_results,
                'top_prediction': formatted_results[0] if formatted_results else None
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)