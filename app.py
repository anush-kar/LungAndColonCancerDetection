# from transformers import pipeline

# model = pipeline("image-classification", model="DunnBC22/vit-base-patch16-224-in21k_lung_and_colon_cancer", use_fast=True)
# response = model("https://storage.googleapis.com/kagglesdsdata/datasets/601280/1079953/lung_colon_image_set/lung_image_sets/lung_aca/lungaca102.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20251024%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251024T151333Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8656da938e88249e9be13cb42d19f15394ef8d2e07d9063eaf25fb29868ae2ed2183156997bca5d6c58110c35c12152f577fcd20198b2998acb229c9596649eb79ad135871762a8bba78d9bfc8748bb46d2a440d6f8bbd9ef08b3ff17c4fa2f31e0eb79eb166ac547c2a8da2eb7a5934e0ea3662e19249b7c14d3c1a2e9929150fe450dc01d09d5d5761dfb05e2a4430d67fa581935fbc73233adc26fc388b819ec4a72b468d62f68f83ca293532d625ac3d523ada86573e44f90aa26292ac7272318891e2e8ab6bcedbe71a990101ca9eb84d233871878975dc2dac8cb57eeece92679d98125ef51584fbfed979cfa95e821bd04ad01d3a06c087ea82e1ff21")

# print(response)

from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import pipeline
from PIL import Image
import os
import json
import csv
import re
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
WAQI_STATIONS_FILE = os.path.join(DATA_DIR, 'waqi_stations_india.txt')
NCRP_INCIDENCE_FILE = os.path.join(DATA_DIR, 'ncrp_lung_cancer_incidence.csv')
RISK_MODEL_ARTIFACT = os.path.join('models', 'patient_risk_selected_features.pkl')
RISK_DATASET_FILE = os.path.join('data', 'cancer patient data sets.csv')

_risk_model_artifact_cache = None
_image_model_cache = None
_incidence_records_cache = None

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
        # device=0 if torch.cuda.is_available() else -1,
        device=-1,
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

def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

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

def normalize_text(value):
    text = str(value or '').lower().strip()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_region_name(region):
    clean = normalize_text(region)
    noise = {
        'district', 'state', 'urban', 'rural', 'and', 'west', 'east',
        'north', 'south', 'upper', 'lower'
    }
    tokens = [t for t in clean.split(' ') if t and t not in noise and len(t) >= 4]
    return list(dict.fromkeys(tokens))

def load_incidence_records():
    global _incidence_records_cache

    if _incidence_records_cache is not None:
        return _incidence_records_cache

    if not os.path.exists(NCRP_INCIDENCE_FILE):
        _incidence_records_cache = []
        return _incidence_records_cache

    records = []
    with open(NCRP_INCIDENCE_FILE, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            region = (row.get('PBCR') or '').strip()
            if not region:
                continue

            record = {
                'region': region,
                'region_norm': normalize_text(region),
                'region_tokens': tokenize_region_name(region),
                'male': {
                    'aar': to_float(row.get('Male_AAR')),
                    'cr': to_float(row.get('Male_CR')),
                    'incidence_n': to_float(row.get('Male_Incidence_n')),
                },
                'female': {
                    'aar': to_float(row.get('Female_AAR')),
                    'cr': to_float(row.get('Female_CR')),
                    'incidence_n': to_float(row.get('Female_Incidence_n')),
                },
            }
            records.append(record)

    _incidence_records_cache = records
    return _incidence_records_cache

def get_incidence_value_for_record(record, metric, gender):
    metric_key = (metric or 'aar').lower()
    gender_key = (gender or 'combined').lower()

    if metric_key not in {'aar', 'cr', 'incidence_n'}:
        raise ValueError('incidence_metric must be aar, cr, or incidence_n')
    if gender_key not in {'male', 'female', 'combined'}:
        raise ValueError('gender must be male, female, or combined')

    male_val = record.get('male', {}).get(metric_key)
    female_val = record.get('female', {}).get(metric_key)

    if gender_key == 'male':
        return male_val
    if gender_key == 'female':
        return female_val

    vals = [v for v in [male_val, female_val] if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))

def find_incidence_region_record(region):
    query = normalize_text(region)
    for rec in load_incidence_records():
        if rec.get('region_norm') == query:
            return rec

    for rec in load_incidence_records():
        if query and query in rec.get('region_norm', ''):
            return rec
    return None

def find_region_stations(region, stations):
    region_norm = normalize_text(region)
    region_tokens = tokenize_region_name(region)
    matches = []

    for station in stations:
        station_norm = normalize_text(station)
        if not station_norm:
            continue

        if region_norm and (region_norm in station_norm or station_norm in region_norm):
            matches.append(station)
            continue

        station_parts = [normalize_text(p) for p in str(station).split(',') if p.strip()]
        station_tokens = []
        for part in station_parts:
            station_tokens.extend([t for t in part.split(' ') if t])
        station_tokens = set(station_tokens)

        if any(token in station_tokens for token in region_tokens):
            matches.append(station)

    return list(dict.fromkeys(matches))

def aggregate_region_waqi(stations, station_cache, waqi_mode, per_region_station_limit):
    chosen = stations[:per_region_station_limit]
    metric_vals = []
    used_stations = []

    for station in chosen:
        if station not in station_cache:
            try:
                data = fetch_waqi_feed(station)
                iaqi = data.get('iaqi', {}) or {}
                station_cache[station] = {
                    'aqi': to_float(data.get('aqi')),
                    'pm25': to_float((iaqi.get('pm25') or {}).get('v')),
                    'pm10': to_float((iaqi.get('pm10') or {}).get('v')),
                }
            except Exception:
                station_cache[station] = None

        snapshot = station_cache.get(station)
        if not snapshot:
            continue

        value = snapshot.get(waqi_mode)
        if value is None:
            continue

        metric_vals.append(value)
        used_stations.append(station)

    if not metric_vals:
        return None, []

    return float(sum(metric_vals) / len(metric_vals)), used_stations

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

def fetch_waqi_feed(query):
    token = WAQI_API_TOKEN or os.environ.get('WAQI_API_TOKEN') or 'demo'
    waqi_url = f"https://api.waqi.info/feed/{quote(query)}/?token={token}"

    with urlopen(waqi_url, timeout=10) as response:
        payload = json.loads(response.read().decode('utf-8'))

    if payload.get('status') != 'ok':
        data = payload.get('data')
        message = data if isinstance(data, str) else 'Unable to fetch data from WAQI API'
        raise ValueError(message)

    return payload.get('data', {})

def collect_waqi_snapshots(stations, limit):
    samples = []
    for station in stations[:limit]:
        try:
            data = fetch_waqi_feed(station)
        except Exception:
            continue

        iaqi = data.get('iaqi', {}) or {}
        row = {
            'station': station,
            'aqi': to_float(data.get('aqi')),
            'pm25': to_float((iaqi.get('pm25') or {}).get('v')),
            'pm10': to_float((iaqi.get('pm10') or {}).get('v')),
        }
        samples.append(row)

    return samples

def build_waqi_mode_vectors(samples, mode):
    vectors = []
    for row in samples:
        aqi = row.get('aqi')
        pm25 = row.get('pm25')
        pm10 = row.get('pm10')

        if mode == 'pm25':
            x_val = pm25
            y_val = aqi
            x_label = 'PM2.5'
            y_label = 'WAQI AQI'
        elif mode == 'pm10':
            x_val = pm10
            y_val = aqi
            x_label = 'PM10'
            y_label = 'WAQI AQI'
        else:
            x_val = pm25
            y_val = pm10
            x_label = 'PM2.5'
            y_label = 'PM10'

        if x_val is None or y_val is None:
            continue

        vectors.append({
            'station': row.get('station'),
            'x': x_val,
            'y': y_val,
        })

    return vectors, x_label, y_label

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

    try:
        data = fetch_waqi_feed(query)
        return jsonify({'success': True, 'data': data})

    except ValueError as ex:
        return jsonify({'error': str(ex)}), 400
    except (HTTPError, URLError):
        return jsonify({'error': 'Failed to connect to WAQI API'}), 502
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/api/waqi/correlation/metadata', methods=['GET'])
def waqi_correlation_metadata():
    stations = load_waqi_stations()
    return jsonify({
        'success': True,
        'data': {
            'data_source': 'Live WAQI station feed',
            'station_count': len(stations),
            'supported_modes': ['pm25', 'pm10', 'waqi_aqi'],
            'supported_methods': ['pearson', 'spearman'],
        }
    })

@app.route('/api/incidence-waqi/metadata', methods=['GET'])
def incidence_waqi_metadata():
    records = load_incidence_records()
    stations = load_waqi_stations()
    return jsonify({
        'success': True,
        'data': {
            'region_count': len(records),
            'regions': sorted([r.get('region') for r in records]),
            'gender_options': ['male', 'female', 'combined'],
            'incidence_metric_options': ['aar', 'cr', 'incidence_n'],
            'waqi_metric_options': ['aqi', 'pm25', 'pm10'],
            'supported_methods': ['pearson', 'spearman'],
            'station_count': len(stations),
            'stations': stations,
        }
    })

@app.route('/api/incidence-waqi/correlation', methods=['POST'])
def incidence_waqi_correlation():
    payload = request.get_json(silent=True) or {}

    selected_region = (payload.get('region') or '').strip()
    gender = (payload.get('gender') or 'combined').lower()
    incidence_metric = (payload.get('incidence_metric') or 'aar').lower()
    waqi_mode = (payload.get('waqi_mode') or 'aqi').lower()
    method = (payload.get('method') or 'pearson').lower()
    requested_stations = payload.get('stations')

    per_region_station_limit = int(payload.get('station_limit', 4) or 4)
    per_region_station_limit = max(1, min(8, per_region_station_limit))

    if gender not in {'male', 'female', 'combined'}:
        return jsonify({'error': 'gender must be male, female, or combined'}), 400
    if incidence_metric not in {'aar', 'cr', 'incidence_n'}:
        return jsonify({'error': 'incidence_metric must be aar, cr, or incidence_n'}), 400
    if waqi_mode not in {'aqi', 'pm25', 'pm10'}:
        return jsonify({'error': 'waqi_mode must be aqi, pm25, or pm10'}), 400
    if method not in {'pearson', 'spearman'}:
        return jsonify({'error': 'method must be pearson or spearman'}), 400

    records = load_incidence_records()
    if len(records) < 3:
        return jsonify({'error': 'Insufficient incidence records for correlation analysis'}), 400

    if isinstance(requested_stations, list):
        stations = [str(s).strip() for s in requested_stations if str(s).strip()]
    else:
        stations = load_waqi_stations()

    if not stations:
        return jsonify({'error': 'No WAQI stations are available'}), 400

    station_cache = {}
    points = []
    skipped_no_station_match = 0
    skipped_missing_data = 0

    for rec in records:
        incidence_value = get_incidence_value_for_record(rec, incidence_metric, gender)
        if incidence_value is None:
            skipped_missing_data += 1
            continue

        region_stations = find_region_stations(rec.get('region'), stations)
        if not region_stations:
            skipped_no_station_match += 1
            continue

        waqi_value, used_stations = aggregate_region_waqi(
            stations=region_stations,
            station_cache=station_cache,
            waqi_mode=waqi_mode,
            per_region_station_limit=per_region_station_limit,
        )

        if waqi_value is None:
            skipped_missing_data += 1
            continue

        points.append({
            'region': rec.get('region'),
            'x': waqi_value,
            'y': incidence_value,
            'stations_used': used_stations,
            'station_count': len(used_stations),
            'is_selected': normalize_text(rec.get('region')) == normalize_text(selected_region),
        })

    if len(points) < 3:
        return jsonify({
            'error': 'Insufficient valid regional points from live WAQI + incidence data. Try selecting different stations or rerun later.'
        }), 400

    xs = [p['x'] for p in points]
    ys = [p['y'] for p in points]

    try:
        r_value, p_value = compute_correlation(xs, ys, method=method)
        slope, intercept = linear_fit(xs, ys)
    except ValueError as ex:
        return jsonify({'error': str(ex)}), 400
    except Exception as ex:
        return jsonify({'error': f'Correlation failed: {str(ex)}'}), 500

    selected_point = None
    if selected_region:
        for p in points:
            if p.get('is_selected'):
                selected_point = p
                break

    metric_label_map = {
        'aar': 'AAR (Age-adjusted rate)',
        'cr': 'CR (Crude rate)',
        'incidence_n': 'Incidence Count',
    }
    waqi_label_map = {
        'aqi': 'WAQI AQI',
        'pm25': 'PM2.5',
        'pm10': 'PM10',
    }

    return jsonify({
        'success': True,
        'correlation': {
            'method': method,
            'r': round(r_value, 4),
            'p_value': (None if p_value is None else round(p_value, 6)),
            'n': len(points),
            'x_metric': waqi_label_map.get(waqi_mode, waqi_mode),
            'y_metric': f"{metric_label_map.get(incidence_metric, incidence_metric)} ({gender})",
            'trendline': {
                'slope': (None if slope is None else round(slope, 6)),
                'intercept': (None if intercept is None else round(intercept, 6))
            },
            'selected_region': selected_region or None,
            'selected_region_found': selected_point is not None,
        },
        'points': points,
        'selected_region_point': selected_point,
        'coverage': {
            'regions_total': len(records),
            'regions_used': len(points),
            'skipped_no_station_match': skipped_no_station_match,
            'skipped_missing_data': skipped_missing_data,
            'station_limit_per_region': per_region_station_limit,
        },
        'interpretation': {
            'direction': 'positive' if r_value >= 0 else 'negative',
            'strength': correlation_strength(abs(r_value)),
            'note': 'Selected region is highlighted for context; correlation is computed across all valid matched regions. Correlation does not imply causation.' if p_value is not None else 'Selected region is highlighted for context; correlation is computed across all valid matched regions. Correlation does not imply causation. p-value unavailable without scipy.'
        }
    })

@app.route('/api/waqi/correlation', methods=['POST'])
def waqi_correlation():
    payload = request.get_json(silent=True) or {}
    mode = (payload.get('mode') or 'pm25').lower()
    method = payload.get('method', 'pearson').lower()
    limit = int(payload.get('limit', 20) or 20)
    limit = max(3, min(limit, 40))
    requested_stations = payload.get('stations')

    if mode not in {'pm25', 'pm10', 'waqi_aqi'}:
        return jsonify({'error': 'mode must be pm25, pm10, or waqi_aqi'}), 400
    if method not in {'pearson', 'spearman'}:
        return jsonify({'error': 'method must be pearson or spearman'}), 400

    if isinstance(requested_stations, list):
        stations = [str(s).strip() for s in requested_stations if str(s).strip()]
    else:
        stations = load_waqi_stations()

    if len(stations) < 3:
        return jsonify({'error': 'At least 3 stations are required for WAQI correlation.'}), 400

    samples = collect_waqi_snapshots(stations, limit=limit)
    vectors, x_label, y_label = build_waqi_mode_vectors(samples, mode)

    xs = []
    ys = []
    points = []
    for item in vectors:
        x_val = item['x']
        y_val = item['y']
        xs.append(x_val)
        ys.append(y_val)
        points.append({
            'station': item.get('station'),
            'x': x_val,
            'y': y_val
        })

    if len(xs) < 3:
        return jsonify({
            'error': 'Insufficient valid live WAQI points. Try increasing station limit or choose different stations.'
        }), 400

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
            'mode': mode,
            'r': round(r_value, 4),
            'p_value': (None if p_value is None else round(p_value, 6)),
            'n': len(xs),
            'x_metric': x_label,
            'y_metric': y_label,
            'stations_requested': min(limit, len(stations)),
            'stations_used': len(xs),
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