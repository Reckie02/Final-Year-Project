  let authenticated = false;
  let selectedFile  = null;

  // ── Login ─────────────────────────────────────────────────────
  async function doLogin() {
    const pw  = document.getElementById('pwInput').value;
    const err = document.getElementById('loginErr');
    const res = await fetch('/api/admin/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ password: pw })
    });
    const data = await res.json();
    if (data.success) {
      authenticated = true;
      document.getElementById('loginGate').style.display    = 'none';
      document.getElementById('adminContent').style.display = 'block';
      loadHealth();
    } else {
      err.textContent = '✗ Incorrect password. Try again.';
      document.getElementById('pwInput').value = '';
    }
  }

  // ── Data health ───────────────────────────────────────────────
  async function loadHealth() {
    const d = await fetch('/api/admin/health').then(r=>r.json());

    const nullClass = d.null_pct < 5 ? 'good' : d.null_pct < 15 ? 'warn' : 'bad';

    let nullRows = d.null_detail.map(n =>
      `<tr>
        <td>${n.column}</td>
        <td>${n.nulls.toLocaleString()}</td>
        <td><span style="color:${n.pct>20?'var(--red)':n.pct>10?'var(--orange)':'var(--green)'}">
          ${n.pct}%</span></td>
      </tr>`
    ).join('');

    document.getElementById('healthStats').innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                  gap:12px;margin-bottom:18px;">
        ${stat('Total Rows',    d.total_rows.toLocaleString(),  'blue')}
        ${stat('Columns',       d.columns,                      'blue')}
        ${stat('Provinces',     d.provinces,                    'blue')}
        ${stat('Unique Sources',d.sources.toLocaleString(),     'blue')}
        ${stat('Date From',     d.date_min,                     'blue')}
        ${stat('Date To',       d.date_max,                     'blue')}
        ${stat('Total Nulls',   d.total_nulls.toLocaleString(), nullClass)}
        ${stat('Null Rate',     d.null_pct + '%',               nullClass)}
      </div>
      ${d.null_detail.length > 0 ? `
        <div style="font-size:12px;font-weight:bold;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px;">
          Columns with missing values
        </div>
        <table class="null-table">
          <thead><tr><th>Column</th><th>Missing Count</th><th>% Missing</th></tr></thead>
          <tbody>${nullRows}</tbody>
        </table>` : `<div style="color:var(--green);font-size:13px;">
          ✅ No missing values found in dataset.</div>`}
    `;
  }

  function stat(label, value, cls) {
    const colors = { blue:'var(--blue)', green:'var(--green)',
                     warn:'var(--orange)', bad:'var(--red)', good:'var(--green)' };
    return `<div style="background:var(--bg);border-radius:8px;padding:12px;">
      <div style="font-size:18px;font-weight:bold;color:${colors[cls]||colors.blue}">
        ${value}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:2px;
                  text-transform:uppercase;letter-spacing:.3px">${label}</div>
    </div>`;
  }

  // ── File handling ─────────────────────────────────────────────
  function handleFileSelect(input) {
    if (input.files[0]) setFile(input.files[0]);
  }

  function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropZone').classList.remove('drag');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  }

  function setFile(file) {
    selectedFile = file;
    document.getElementById('dropLabel').textContent = `Selected: ${file.name}`;
    document.getElementById('uploadBtn').disabled = false;
  }

  // ── Upload ────────────────────────────────────────────────────
  async function uploadFile() {
    if (!selectedFile) return;
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true; btn.textContent = '⏳ Uploading...';

    const form = new FormData();
    form.append('file', selectedFile);

    try {
      const res  = await fetch('/api/admin/upload', { method:'POST', body:form });
      const data = await res.json();
      const box  = document.getElementById('uploadResult');

      if (data.success) {
        box.className = 'result-box success';
        box.style.display = 'block';
        box.innerHTML = `
          <strong>✅ Upload successful!</strong><br>
          File: ${data.filename}<br>
          Rows loaded: ${data.rows.toLocaleString()}<br>
          Columns: ${data.columns}<br>
          <em>The live dashboard now reflects this new data.</em>`;
        showToast('Dataset uploaded successfully!', 'success');
        loadHealth();
      } else {
        box.className = 'result-box error';
        box.style.display = 'block';
        box.innerHTML = `<strong>✗ Upload failed:</strong> ${data.error}`;
        showToast('Upload failed', 'error');
      }
    } catch(e) {
      showToast('Network error: ' + e.message, 'error');
    }

    btn.disabled = false; btn.textContent = 'Upload Dataset';
  }

  // ── Retrain ───────────────────────────────────────────────────
  async function retrainModel() {
    const btn = document.getElementById('retrainBtn');
    btn.disabled = true; btn.textContent = '⏳ Training... please wait';

    try {
      const res  = await fetch('/api/admin/retrain', { method:'POST' });
      const data = await res.json();
      const box  = document.getElementById('retrainResult');

      if (data.success) {
        box.className = 'result-box success';
        box.style.display = 'block';
        box.innerHTML = `
          <strong>✅ Model retrained successfully!</strong><br>
          Accuracy: <strong>${data.accuracy}%</strong><br>
          Training records: ${data.training_rows.toLocaleString()}<br>
          Testing records: ${data.testing_rows.toLocaleString()}<br>
          <em>The Risk Predictor page now uses this updated model.</em>`;
        showToast(`Model retrained — ${data.accuracy}% accuracy`, 'success');
      } else {
        box.className = 'result-box error';
        box.style.display = 'block';
        box.innerHTML = `<strong>✗ Retraining failed:</strong> ${data.error}`;
        showToast('Retraining failed', 'error');
      }
    } catch(e) {
      showToast('Network error: ' + e.message, 'error');
    }

    btn.disabled = false; btn.textContent = '🔁 Retrain Model Now';
  }

  // ── Toast notification ────────────────────────────────────────
  function showToast(msg, type='success') {
    const t = document.getElementById('toast');
    t.textContent = msg; t.className = `toast ${type} show`;
    setTimeout(() => t.classList.remove('show'), 3500);
  }
