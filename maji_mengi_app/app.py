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