from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv('master_clean.csv')
df['time_of_record'] = pd.to_datetime(df['time_of_record'], errors='coerce')
df['month'] = df['time_of_record'].dt.month_name()
df['hour']  = df['time_of_record'].dt.hour
print(f"✅ Data loaded: {len(df):,} records")

# ── Load model ─────────────────────────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    enc = pickle.load(f)
print("✅ Model loaded")


# ══════════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')


# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════
def apply_filters(province='All', month='All', hour='All'):
    out = df.copy()
    if province != 'All':
        out = out[out['province_name'] == province]
    if month != 'All':
        out = out[out['month'] == month]
    if hour != 'All':
        out = out[out['hour'] == int(hour)]
    return out


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD APIs
# ══════════════════════════════════════════════════════════════════
@app.route('/api/filters')
def get_filters():
    provinces = sorted(df['province_name'].dropna().unique().tolist())
    months_order = ['January','February','March','April','May','June',
                    'July','August','September','October','November','December']
    months = [m for m in months_order if m in df['month'].unique()]
    return jsonify({'provinces': provinces, 'months': months})


@app.route('/api/summary')
def summary():
    filtered = apply_filters(
        request.args.get('province','All'),
        request.args.get('month','All'),
        request.args.get('hour','All')
    )
    total      = len(filtered)
    func       = (filtered['functionality_status'] == 'functional').sum()
    non_func   = (filtered['functionality_status'] == 'non_functional').sum()
    partial    = (filtered['functionality_status'] == 'partially_functional').sum()
    avg_q      = filtered['time_in_queue'].mean()
    people     = filtered['number_of_people_served'].sum()

    return jsonify({
        'total':          int(total),
        'functional':     int(func),
        'non_functional': int(non_func),
        'partial':        int(partial),
        'avg_queue':      round(float(avg_q), 1) if pd.notna(avg_q) else 0,
        'people_served':  int(people) if pd.notna(people) else 0,
        'func_pct':       round(func/total*100, 1) if total else 0,
        'nonfunc_pct':    round(non_func/total*100, 1) if total else 0,
    })


@app.route('/api/status_breakdown')
def status_breakdown():
    filtered = apply_filters(
        request.args.get('province','All'),
        request.args.get('month','All')
    )
    return jsonify(filtered['functionality_status'].value_counts().to_dict())


@app.route('/api/source_types')
def source_types():
    filtered = apply_filters(
        request.args.get('province','All'),
        request.args.get('month','All')
    )
    return jsonify(filtered['type_of_water_source'].value_counts().to_dict())


@app.route('/api/queue_by_hour')
def queue_by_hour():
    filtered = apply_filters(province=request.args.get('province','All'))
    hourly = (filtered.groupby('hour')['time_in_queue']
              .mean().round(1).dropna().reset_index())
    return jsonify({
        'hours':  hourly['hour'].tolist(),
        'queues': hourly['time_in_queue'].tolist()
    })


@app.route('/api/province_comparison')
def province_comparison():
    filtered = apply_filters(month=request.args.get('month','All'))
    result = (filtered.groupby('province_name').agg(
        avg_queue     =('time_in_queue',        'mean'),
        non_func_count=('functionality_status', lambda x: (x=='non_functional').sum()),
        total         =('functionality_status', 'count'),
        people_served =('number_of_people_served','sum')
    ).reset_index())
    result['non_func_pct'] = (result['non_func_count']/result['total']*100).round(1)
    result['avg_queue']    = result['avg_queue'].round(1)
    result['people_served']= result['people_served'].astype(int)
    return jsonify(result.to_dict(orient='records'))


@app.route('/api/trend_by_month')
def trend_by_month():
    province = request.args.get('province','All')
    filtered = apply_filters(province=province)
    months_order = ['January','February','March','April','May','June',
                    'July','August','September','October','November','December']
    trend = (filtered.groupby('month').agg(
        non_func_count=('functionality_status', lambda x: (x=='non_functional').sum()),
        total         =('functionality_status', 'count'),
        avg_queue     =('time_in_queue',        'mean')
    ).reset_index())
    trend['non_func_pct'] = (trend['non_func_count']/trend['total']*100).round(1)
    trend['avg_queue']    = trend['avg_queue'].round(1)
    # Sort by calendar month order
    trend['month_order'] = trend['month'].apply(
        lambda m: months_order.index(m) if m in months_order else 99
    )
    trend = trend.sort_values('month_order').drop('month_order', axis=1)
    return jsonify(trend.to_dict(orient='records'))


# ══════════════════════════════════════════════════════════════════
#  PREDICTION API
# ══════════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        source_type   = data['source_type']
        people_served = float(data['people_served'])
        queue_time    = float(data['queue_time'])
        quality_score = float(data['quality_score'])
        location_type = data['location_type']

        # Encode inputs using saved encoders
        type_enc = enc['le_type'].transform([source_type])[0]
        loc_enc  = enc['le_loc'].transform([location_type])[0]

        features = np.array([[type_enc, people_served, queue_time,
                               quality_score, loc_enc]])

        proba             = model.predict_proba(features)[0]
        failure_idx       = list(enc['le_status'].classes_).index('non_functional')
        failure_prob      = round(float(proba[failure_idx]) * 100, 1)

        if failure_prob >= 66:
            risk = 'High Risk'
            color = '#e74c3c'
            advice = 'Immediate inspection and maintenance required. This water point is likely failing.'
        elif failure_prob >= 33:
            risk = 'Medium Risk'
            color = '#f39c12'
            advice = 'Schedule maintenance within the next 30 days. Monitor queue times closely.'
        else:
            risk = 'Low Risk'
            color = '#27ae60'
            advice = 'Water point appears stable. Continue regular monitoring schedule.'

        # All class probabilities for the chart
        class_probs = {
            cls: round(float(p)*100, 1)
            for cls, p in zip(enc['le_status'].classes_, proba)
        }

        return jsonify({
            'failure_probability': failure_prob,
            'risk_level':  risk,
            'risk_color':  color,
            'advice':      advice,
            'all_probs':   class_probs
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ══════════════════════════════════════════════════════════════════
#  VALID OPTIONS (for prediction form dropdowns)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/options')
def options():
    return jsonify({
        'source_types':   list(enc['le_type'].classes_),
        'location_types': list(enc['le_loc'].classes_)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
    
#=========================================================================
 # ROuTeS
#========================================================================= 
    import os
import io
from werkzeug.utils import secure_filename

ADMIN_PASSWORD = "majimengi2025"   # change this to anything you want
UPLOAD_FOLDER  = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Admin page ─────────────────────────────────────────────────
@app.route('/admin')
def admin_page():
    return render_template('admin.html')

# ── Auth check ─────────────────────────────────────────────────
@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    if data.get('password') == ADMIN_PASSWORD:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Incorrect password'}), 401

# ── Data health stats ──────────────────────────────────────────
@app.route('/api/admin/health')
def data_health():
    total_rows   = len(df)
    null_counts  = df.isnull().sum()
    total_nulls  = int(null_counts.sum())
    date_min     = str(df['time_of_record'].min())[:10] if 'time_of_record' in df.columns else 'N/A'
    date_max     = str(df['time_of_record'].max())[:10] if 'time_of_record' in df.columns else 'N/A'
    provinces    = int(df['province_name'].nunique())
    sources      = int(df['source_id'].nunique()) if 'source_id' in df.columns else 0

    top_nulls = (null_counts[null_counts > 0]
                 .sort_values(ascending=False)
                 .head(6)
                 .reset_index())
    top_nulls.columns = ['column', 'nulls']
    top_nulls['pct'] = (top_nulls['nulls'] / total_rows * 100).round(1)

    return jsonify({
        'total_rows':   total_rows,
        'total_nulls':  total_nulls,
        'null_pct':     round(total_nulls / (total_rows * len(df.columns)) * 100, 2),
        'date_min':     date_min,
        'date_max':     date_max,
        'provinces':    provinces,
        'sources':      sources,
        'columns':      len(df.columns),
        'null_detail':  top_nulls.to_dict(orient='records')
    })

# ── Upload new CSV ─────────────────────────────────────────────
@app.route('/api/admin/upload', methods=['POST'])
def upload_csv():
    global df

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are accepted'}), 400

    try:
        content   = file.read()
        new_df    = pd.read_csv(io.BytesIO(content))
        row_count = len(new_df)
        cols      = list(new_df.columns)

        # Validate required columns exist
        required = ['source_id', 'functionality_status',
                    'type_of_water_source', 'number_of_people_served']
        missing  = [c for c in required if c not in cols]
        if missing:
            return jsonify({
                'error': f'Missing required columns: {missing}'
            }), 400

        # Save to disk and reload into memory
        filename = secure_filename(file.filename)
        saved_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(saved_path, 'wb') as f_out:
            f_out.write(content)

        new_df['time_of_record'] = pd.to_datetime(
            new_df['time_of_record'], errors='coerce')
        new_df['month'] = new_df['time_of_record'].dt.month_name()
        new_df['hour']  = new_df['time_of_record'].dt.hour
        df = new_df   # replace the global dataframe

        # Also overwrite master_clean.csv
        df.to_csv('master_clean.csv', index=False)

        return jsonify({
            'success':   True,
            'rows':      row_count,
            'columns':   len(cols),
            'filename':  filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Retrain model on current data ─────────────────────────────
@app.route('/api/admin/retrain', methods=['POST'])
def retrain():
    global model, enc
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    try:
        model_df = df[df['functionality_status'] != 'unknown'][[
            'type_of_water_source', 'number_of_people_served',
            'time_in_queue', 'subjective_quality_score',
            'location_type', 'functionality_status'
        ]].dropna().copy()

        le_type   = LabelEncoder()
        le_loc    = LabelEncoder()
        le_status = LabelEncoder()

        model_df['type_encoded']   = le_type.fit_transform(model_df['type_of_water_source'])
        model_df['loc_encoded']    = le_loc.fit_transform(model_df['location_type'])
        model_df['status_encoded'] = le_status.fit_transform(model_df['functionality_status'])

        X = model_df[['type_encoded', 'number_of_people_served',
                       'time_in_queue', 'subjective_quality_score', 'loc_encoded']]
        y = model_df['status_encoded']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        new_model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, class_weight='balanced')
        new_model.fit(X_train, y_train)

        acc = round(accuracy_score(y_test, new_model.predict(X_test)) * 100, 2)

        # Save new model to disk
        import pickle
        with open('model.pkl', 'wb') as f:
            pickle.dump(new_model, f)
        with open('encoders.pkl', 'wb') as f:
            pickle.dump({'le_type': le_type,
                         'le_loc':  le_loc, 'le_status': le_status}, f)

        model = new_model
        enc   = {'le_type': le_type, 'le_loc': le_loc, 'le_status': le_status}

        return jsonify({
            'success':       True,
            'accuracy':      acc,
            'training_rows': len(X_train),
            'testing_rows':  len(X_test)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500