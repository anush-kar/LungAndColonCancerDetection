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
import math
from dotenv import load_dotenv
from pathlib import Path

from risk_model import ensure_risk_model, predict_patient_likelihood

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

load_dotenv()
WAQI_API_TOKEN = os.getenv('WAQI_API_KEY')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MICROPLASTICS_DATA_FILE = os.path.join(DATA_DIR, 'microplastics_regions.json')
WAQI_STATIONS_FILE = os.path.join(DATA_DIR, 'waqi_stations_india.txt')
RISK_MODEL_ARTIFACT = os.path.join('models', 'patient_risk_selected_features.pkl')
RISK_DATASET_FILE = os.path.join('data', 'cancer patient data sets.csv')

_risk_model_artifact_cache = None
_image_model_cache = None

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_image_model():
    global _image_model_cache

    if _image_model_cache is not None:
        return _image_model_cache

    print("Loading model...")
    _image_model_cache = pipeline(
        "image-classification",
        model="DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer",
        device=0 if torch.cuda.is_available() else -1,
        use_fast=True,
    )
    print("Model loaded successfully!")
    return _image_model_cache

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

def load_microplastics_data():
    if not os.path.exists(MICROPLASTICS_DATA_FILE):
        return []

    with open(MICROPLASTICS_DATA_FILE, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload if isinstance(payload, list) else []

def find_region_record(region):
    records = load_microplastics_data()
    if not records:
        return None

    region_l = region.lower().strip()
    exact = [r for r in records if str(r.get('region', '')).lower().strip() == region_l]
    if exact:
        return sorted(exact, key=lambda x: x.get('year', 0), reverse=True)[0]

    partial = [r for r in records if region_l in str(r.get('region', '')).lower()]
    if partial:
        return sorted(partial, key=lambda x: x.get('year', 0), reverse=True)[0]

    return None

def get_incidence_value(record, cancer_type):
    ct = (cancer_type or 'combined').lower()
    aca = float(record.get('lung_aca_incidence', 0))
    scc = float(record.get('lung_scc_incidence', 0))

    if ct in {'aca', 'lung_aca', 'adenocarcinoma'}:
        return aca
    if ct in {'scc', 'lung_scc', 'squamous'}:
        return scc
    return (aca + scc) / 2.0

def clamp01(value):
    return max(0.0, min(1.0, value))

def normalize_value(value, low, high):
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))

def correlation_strength(abs_r):
    if abs_r >= 0.8:
        return 'very strong'
    if abs_r >= 0.6:
        return 'strong'
    if abs_r >= 0.4:
        return 'moderate'
    if abs_r >= 0.2:
        return 'weak'
    return 'very weak'

def pearson_corr_py(xs, ys):
    n = len(xs)
    if n != len(ys) or n < 3:
        raise ValueError('At least 3 valid regional records are required for correlation.')

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]

    numerator = sum(a * b for a, b in zip(dx, dy))
    denom_x = math.sqrt(sum(a * a for a in dx))
    denom_y = math.sqrt(sum(b * b for b in dy))
    denominator = denom_x * denom_y

    if denominator == 0:
        raise ValueError('Correlation is undefined because variance is zero in one series.')

    return float(numerator / denominator)

def rank_data(values):
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks

def linear_fit(xs, ys):
    n = len(xs)
    if n < 2:
        return None, None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None, None

    slope = num / den
    intercept = mean_y - slope * mean_x
    return float(slope), float(intercept)

def compute_correlation(xs, ys, method='pearson'):
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError('At least 3 valid regional records are required for correlation.')

    if method == 'spearman':
        if spearmanr is not None:
            r, p_value = spearmanr(xs, ys)
        else:
            rx = rank_data(xs)
            ry = rank_data(ys)
            r = pearson_corr_py(rx, ry)
            p_value = None
    else:
        if pearsonr is not None:
            r, p_value = pearsonr(xs, ys)
        else:
            r = pearson_corr_py(xs, ys)
            p_value = None

    return float(r), (None if p_value is None else float(p_value))


def get_patient_risk_model_artifact():
    global _risk_model_artifact_cache

    if _risk_model_artifact_cache is not None:
        return _risk_model_artifact_cache

    project_root = Path(__file__).parent
    artifact_path = project_root / RISK_MODEL_ARTIFACT
    dataset_path = project_root / RISK_DATASET_FILE

    _risk_model_artifact_cache = ensure_risk_model(
        artifact_path=artifact_path,
        csv_path=dataset_path,
    )
    return _risk_model_artifact_cache

def load_waqi_stations():
    if not os.path.exists(WAQI_STATIONS_FILE):
        return []

    stations = []
    with open(WAQI_STATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                stations.append(name)

    # Preserve order, remove duplicates
    deduped = list(dict.fromkeys(stations))
    return deduped

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cities.js')
def cities_js():
    return send_from_directory('templates', 'cities.js')

@app.route('/style.css')
def style_css():
    return send_from_directory('templates', 'style.css')

@app.route('/microplastics.js')
def microplastics_js():
    return send_from_directory('templates', 'microplastics.js')

@app.route('/api/waqi-stations', methods=['GET'])
def waqi_stations():
    stations = load_waqi_stations()
    return jsonify({
        'success': True,
        'data': {
            'count': len(stations),
            'stations': stations
        }
    })

@app.route('/api/air-quality', methods=['GET'])
def get_air_quality():
    station = request.args.get('station', '').strip()
    city = request.args.get('city', '').strip()
    query = station or city
    if not query:
        return jsonify({'error': 'Station name is required'}), 400

    token = WAQI_API_TOKEN or os.environ.get('WAQI_API_TOKEN') or 'demo'
    waqi_url = f"https://api.waqi.info/feed/{quote(query)}/?token={token}"

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

@app.route('/api/microplastics/metadata', methods=['GET'])
def microplastics_metadata():
    records = load_microplastics_data()
    regions = sorted({r.get('region') for r in records if r.get('region')})
    years = sorted({r.get('year') for r in records if r.get('year')})
    return jsonify({
        'success': True,
        'data': {
            'data_source': 'Regional environmental + epidemiological baseline dataset',
            'record_count': len(records),
            'regions': regions,
            'years': years,
            'supported_cancer_types': ['combined', 'lung_aca', 'lung_scc'],
            'supported_methods': ['pearson', 'spearman']
        }
    })

@app.route('/api/microplastics/exposure', methods=['GET'])
def get_microplastics_exposure():
    region = request.args.get('region', '').strip()
    if not region:
        return jsonify({'error': 'Region is required'}), 400

    record = find_region_record(region)
    if not record:
        return jsonify({'error': f'No microplastics dataset found for region: {region}'}), 404

    return jsonify({'success': True, 'data': record})

@app.route('/api/microplastics/correlation', methods=['POST'])
def microplastics_correlation():
    payload = request.get_json(silent=True) or {}
    cancer_type = payload.get('cancer_type', 'combined')
    method = payload.get('method', 'pearson').lower()

    if method not in {'pearson', 'spearman'}:
        return jsonify({'error': 'method must be pearson or spearman'}), 400

    records = load_microplastics_data()
    if len(records) < 3:
        return jsonify({'error': 'Insufficient records for correlation analysis'}), 400

    xs = []
    ys = []
    points = []
    for r in records:
        micro = r.get('microplastics_air')
        if micro is None:
            continue
        x_val = float(micro)
        y_val = get_incidence_value(r, cancer_type)
        xs.append(x_val)
        ys.append(y_val)
        points.append({
            'region': r.get('region'),
            'year': r.get('year'),
            'x': x_val,
            'y': y_val
        })

    try:
        r_value, p_value = compute_correlation(xs, ys, method=method)
        slope, intercept = linear_fit(xs, ys)
    except ValueError as ex:
        return jsonify({'error': str(ex)}), 400
    except Exception as ex:
        return jsonify({'error': f'Correlation failed: {str(ex)}'}), 500

    return jsonify({
        'success': True,
        'correlation': {
            'method': method,
            'r': round(r_value, 4),
            'p_value': (None if p_value is None else round(p_value, 6)),
            'n': len(xs),
            'target': cancer_type,
            'trendline': {
                'slope': (None if slope is None else round(slope, 6)),
                'intercept': (None if intercept is None else round(intercept, 6))
            }
        },
        'points': points,
        'interpretation': {
            'direction': 'positive' if r_value >= 0 else 'negative',
            'strength': correlation_strength(abs(r_value)),
            'note': 'Correlation does not imply causation.' if p_value is not None else 'Correlation does not imply causation. p-value unavailable without scipy.'
        }
    })

@app.route('/api/risk/enhanced', methods=['POST'])
def enhanced_risk():
    payload = request.get_json(silent=True) or {}
    region = (payload.get('region') or '').strip()
    cancer_type = payload.get('cancer_type', 'combined')
    image_score = payload.get('image_score', None)

    if not region:
        return jsonify({'error': 'region is required'}), 400
    if image_score is None:
        return jsonify({'error': 'image_score is required'}), 400

    try:
        image_score = float(image_score)
    except ValueError:
        return jsonify({'error': 'image_score must be a number'}), 400

    image_risk = image_score / 100.0 if image_score > 1 else image_score
    image_risk = clamp01(image_risk)

    record = find_region_record(region)
    if not record:
        return jsonify({'error': f'No microplastics dataset found for region: {region}'}), 404

    all_records = load_microplastics_data()
    micro_values = [float(r.get('microplastics_air', 0)) for r in all_records]
    pm25_values = [float(r.get('pm25', 0)) for r in all_records]

    micro_norm = normalize_value(float(record.get('microplastics_air', 0)), min(micro_values), max(micro_values))
    pm25_norm = normalize_value(float(record.get('pm25', 0)), min(pm25_values), max(pm25_values))
    incidence_norm = normalize_value(get_incidence_value(record, cancer_type),
                                     min(get_incidence_value(r, cancer_type) for r in all_records),
                                     max(get_incidence_value(r, cancer_type) for r in all_records))

    environmental_index = clamp01((0.45 * micro_norm) + (0.35 * pm25_norm) + (0.20 * incidence_norm))
    combined_risk = clamp01((0.70 * image_risk) + (0.30 * environmental_index))
    uncertainty = clamp01(0.08 + (0.15 * (1 - len(all_records) / max(len(all_records), 20))))

    if combined_risk >= 0.75:
        tier = 'high'
    elif combined_risk >= 0.45:
        tier = 'moderate'
    else:
        tier = 'low'

    explanation = [
        f"Image model contributes {(0.70 * image_risk * 100):.1f} risk points.",
        f"Environmental exposure contributes {(0.30 * environmental_index * 100):.1f} risk points.",
        f"Region matched: {record.get('region')} ({record.get('year')}).",
        "This is a decision-support estimate and not a diagnosis."
    ]

    return jsonify({
        'success': True,
        'risk': {
            'base_image_risk': round(image_risk, 4),
            'environmental_index': round(environmental_index, 4),
            'combined_risk': round(combined_risk, 4),
            'risk_tier': tier,
            'uncertainty': round(uncertainty, 4),
            'explanation': explanation,
            'region_data': {
                'microplastics_air': record.get('microplastics_air'),
                'pm25': record.get('pm25'),
                'pm10': record.get('pm10')
            }
        }
    })

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
            
            # Get predictions (lazy-load model on first request)
            image_model = get_image_model()
            results = image_model(image)
            
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


@app.route('/api/risk/patient-likelihood', methods=['POST'])
def predict_patient_risk_likelihood():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'Request body must be a valid JSON object.'}), 400

    try:
        artifact = get_patient_risk_model_artifact()
        result = predict_patient_likelihood(
            payload=payload,
            artifact=artifact,
            model_name=payload.get('model_name'),
        )
    except ValueError as ex:
        return jsonify({'error': str(ex)}), 400
    except Exception as ex:
        return jsonify({'error': f'Patient risk prediction failed: {str(ex)}'}), 500

    return jsonify({'success': True, 'data': result})


@app.route('/api/risk/patient-likelihood/metadata', methods=['GET'])
def patient_risk_prediction_metadata():
    try:
        artifact = get_patient_risk_model_artifact()
        return jsonify({
            'success': True,
            'data': {
                'expected_features': artifact.get('feature_columns', []),
                'classes': artifact.get('class_labels', {}),
                'available_models': artifact.get('available_models', []),
                'default_model': artifact.get('default_model_name'),
                'metrics': artifact.get('metrics', {}),
            },
        })
    except Exception as ex:
        return jsonify({'error': f'Unable to load risk model metadata: {str(ex)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)