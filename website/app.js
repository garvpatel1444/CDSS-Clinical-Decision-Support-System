// ============================================================
//  CDSS — Clinical Decision Support System
//  app.js — API calls, gauge, history, download report
// ============================================================

const API_BASE = 'http://localhost:8000';

// ============================================================
//  ADVICE DATA
// ============================================================

const ADVICE = {
  heart: {
    'Low Risk': {
      diet: [
        'Eat plenty of fruits, vegetables, and whole grains',
        'Choose lean proteins like fish, chicken, and legumes',
        'Limit saturated fats — use olive oil instead of butter',
        'Reduce sodium intake to under 2300 mg/day',
        'Stay hydrated — drink 8–10 glasses of water daily',
      ],
      exercise: [
        '150 minutes of moderate aerobic activity per week (e.g. brisk walking)',
        'Strength training 2 times per week',
        'Yoga or stretching for flexibility',
        'Take stairs instead of lifts',
        'Aim for 8,000–10,000 steps per day',
      ]
    },
    'Medium Risk': {
      diet: [
        'Follow a Mediterranean or DASH diet strictly',
        'Reduce red meat consumption to once a week',
        'Avoid processed and packaged foods high in trans fats',
        'Increase omega-3 intake: salmon, walnuts, flaxseeds',
        'Limit alcohol to one drink per day maximum',
        'Eat smaller, more frequent meals to control blood pressure',
      ],
      exercise: [
        '30 minutes of brisk walking or cycling, 5 days a week',
        'Low-impact swimming is excellent for heart health',
        'Avoid high-intensity workouts without medical clearance',
        'Monitor heart rate during exercise — keep under 70% max HR',
        'Try cardiac rehabilitation programs if available',
      ]
    },
    'High Risk': {
      diet: [
        '⚠ Immediately restrict sodium to under 1500 mg/day',
        'Completely avoid fried foods, fast food, and processed meats',
        'Eat heart-protective foods: oats, berries, leafy greens, fish',
        'Avoid caffeine and alcohol until medically cleared',
        'Follow a cardiologist-prescribed diet plan',
        'Consume soluble fibre (oats, beans) to reduce LDL cholesterol',
      ],
      exercise: [
        '⚠ Consult your cardiologist before any exercise',
        'Light walking (10–15 mins) only if medically approved',
        'Avoid strenuous or high-intensity activities',
        'Rest adequately — overexertion is dangerous',
        'Cardiac rehab under supervised medical care is recommended',
      ]
    }
  },
  diabetes: {
    'Low Risk': {
      diet: [
        'Maintain a balanced diet rich in whole grains and fibre',
        'Limit refined carbohydrates and sugary drinks',
        'Include plenty of non-starchy vegetables in every meal',
        'Choose low glycaemic index (GI) foods',
        'Eat regular meals — avoid skipping breakfast',
      ],
      exercise: [
        '150 minutes of moderate exercise per week',
        'Walking after meals helps control blood sugar',
        'Cycling, swimming, or dancing are great options',
        'Resistance training helps improve insulin sensitivity',
        'Reduce sedentary time — stand or walk every hour',
      ]
    },
    'Medium Risk': {
      diet: [
        'Follow a diabetic-friendly low-GI diet',
        'Avoid white rice, white bread, and sugary snacks',
        'Include protein in every meal to slow glucose absorption',
        'Choose brown rice, quinoa, and whole wheat alternatives',
        'Monitor carbohydrate portions — aim for 45–60g per meal',
        'Include cinnamon and fenugreek — known to improve insulin sensitivity',
      ],
      exercise: [
        '30 minutes of moderate aerobic exercise daily',
        'Post-meal walks of 10–15 minutes are highly effective',
        'Strength training 3 days a week to improve glucose uptake',
        'Monitor blood sugar before and after exercise',
        'Yoga and stress reduction significantly help diabetes management',
      ]
    },
    'High Risk': {
      diet: [
        '⚠ Consult an endocrinologist and dietitian immediately',
        'Strictly limit carbohydrates to under 130g/day',
        'Avoid all sugary foods, beverages, and fruit juices',
        'Eat small, frequent meals every 3–4 hours',
        'Prioritise high-fibre, low-GI vegetables and lean proteins',
        'Track all food with a glucose diary or app',
      ],
      exercise: [
        '⚠ Exercise under medical supervision',
        'Light to moderate walking 20–30 mins after meals',
        'Avoid exercising on an empty stomach — risk of hypoglycaemia',
        'Check blood sugar before and after every session',
        'Strength training can significantly reduce insulin resistance',
      ]
    }
  }
};

// ============================================================
//  GAUGE DRAWING
// ============================================================

function drawGauge(canvasId, probability) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = canvas.width / 2;
  const cy = canvas.height - 10;
  const r = 90;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background arc
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 16;
  ctx.stroke();

  // Colour based on probability
  let color = '#16a34a';
  if (probability >= 0.7) color = '#dc2626';
  else if (probability >= 0.3) color = '#ca8a04';

  // Filled arc
  const angle = Math.PI + (probability * Math.PI);
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.strokeStyle = color;
  ctx.lineWidth = 16;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Needle
  const needleAngle = Math.PI + (probability * Math.PI);
  const nx = cx + (r - 20) * Math.cos(needleAngle);
  const ny = cy + (r - 20) * Math.sin(needleAngle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Centre dot
  ctx.beginPath();
  ctx.arc(cx, cy, 6, 0, Math.PI * 2);
  ctx.fillStyle = '#1e293b';
  ctx.fill();

  // Labels
  ctx.font = '11px DM Sans, sans-serif';
  ctx.fillStyle = '#94a3b8';
  ctx.textAlign = 'left';
  ctx.fillText('0.0', cx - r - 8, cy + 16);
  ctx.textAlign = 'center';
  ctx.fillText('0.5', cx, cy - r - 10);
  ctx.textAlign = 'right';
  ctx.fillText('1.0', cx + r + 8, cy + 16);
}

// ============================================================
//  RISK BADGE COLOUR
// ============================================================

function applyRiskBadge(badgeEl, risk) {
  badgeEl.textContent = risk;
  badgeEl.className = 'result-badge';
  if (risk === 'Low Risk') badgeEl.classList.add('low');
  else if (risk === 'Medium Risk') badgeEl.classList.add('medium');
  else badgeEl.classList.add('high');
}

// ============================================================
//  ADVICE RENDERER
// ============================================================

function renderAdvice(containerId, type, risk) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = ADVICE[type][risk];
  container.innerHTML = `
    <div class="advice-block">
      <h4>🥗 Diet Recommendations</h4>
      <ul>${data.diet.map(d => `<li>${d}</li>`).join('')}</ul>
    </div>
    <div class="advice-block">
      <h4>🏃 Exercise Recommendations</h4>
      <ul>${data.exercise.map(e => `<li>${e}</li>`).join('')}</ul>
    </div>
  `;
}

// ============================================================
//  HEART PREDICTION
// ============================================================

async function predictHeart() {
  const fields = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
  const values = {};
  for (const f of fields) {
    const el = document.getElementById(f);
    if (!el || el.value === '') {
      showError('heart-error', `Please fill in all fields. Missing: ${f}`);
      return;
    }
    values[f] = parseFloat(el.value);
  }

  hideError('heart-error');
  showLoading();

  try {
    const res = await fetch(`${API_BASE}/predict/heart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'API error');
    }

    const data = await res.json();
    hideLoading();
    displayHeartResult(data, values);
    saveToHistory('heart', data, values);

  } catch (e) {
    hideLoading();
    showError('heart-error', `Error: ${e.message}. Make sure the API is running at ${API_BASE}`);
  }
}

function displayHeartResult(data, inputs) {
  const panel = document.getElementById('heart-result');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  applyRiskBadge(document.getElementById('heart-risk-badge'), data.risk_level);

  document.getElementById('heart-prob-text').textContent = data.probability.toFixed(2);
  document.getElementById('heart-risk-level').textContent = data.risk_level;
  document.getElementById('heart-prediction').textContent = data.prediction === 1 ? 'Disease Detected' : 'No Disease Detected';
  document.getElementById('heart-recommendation').textContent = data.recommendation;

  drawGauge('heart-gauge', data.probability);
  renderAdvice('heart-advice', 'heart', data.risk_level);

  // Store last result for download
  window._lastHeartResult = { data, inputs };
}

// ============================================================
//  DIABETES PREDICTION
// ============================================================

async function predictDiabetes() {
  const fields = ['pregnancies','glucose','blood_pressure','skin_thickness','insulin','bmi','diabetes_pedigree_function','age'];
  const values = {};
  for (const f of fields) {
    const el = document.getElementById(f);
    if (!el || el.value === '') {
      showError('diabetes-error', `Please fill in all fields. Missing: ${f}`);
      return;
    }
    values[f] = parseFloat(el.value);
  }

  hideError('diabetes-error');
  showLoading();

  try {
    const res = await fetch(`${API_BASE}/predict/diabetes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values)
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'API error');
    }

    const data = await res.json();
    hideLoading();
    displayDiabetesResult(data, values);
    saveToHistory('diabetes', data, values);

  } catch (e) {
    hideLoading();
    showError('diabetes-error', `Error: ${e.message}. Make sure the API is running at ${API_BASE}`);
  }
}

function displayDiabetesResult(data, inputs) {
  const panel = document.getElementById('diabetes-result');
  panel.style.display = 'block';
  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  applyRiskBadge(document.getElementById('diabetes-risk-badge'), data.risk_level);

  document.getElementById('diabetes-prob-text').textContent = data.probability.toFixed(2);
  document.getElementById('diabetes-risk-level').textContent = data.risk_level;
  document.getElementById('diabetes-prediction').textContent = data.prediction === 1 ? 'Diabetic' : 'Non-Diabetic';
  document.getElementById('diabetes-recommendation').textContent = data.recommendation;

  drawGauge('diabetes-gauge', data.probability);
  renderAdvice('diabetes-advice', 'diabetes', data.risk_level);

  window._lastDiabetesResult = { data, inputs };
}

// ============================================================
//  HISTORY (localStorage)
// ============================================================

function saveToHistory(type, result, inputs) {
  const history = getHistory();
  const entry = {
    id: Date.now(),
    type,
    timestamp: new Date().toLocaleString(),
    risk_level: result.risk_level,
    probability: result.probability,
    prediction: result.prediction,
    recommendation: result.recommendation,
    inputs
  };
  history.unshift(entry);
  localStorage.setItem('cdss_history', JSON.stringify(history));
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem('cdss_history') || '[]');
  } catch { return []; }
}

function updateSummary() {
  const history = getHistory();
  const heartCount = history.filter(h => h.type === 'heart').length;
  const diabetesCount = history.filter(h => h.type === 'diabetes').length;
  const highCount = history.filter(h => h.risk_level === 'High Risk').length;

  const tc = document.getElementById('total-count');
  const hc = document.getElementById('heart-count');
  const dc = document.getElementById('diabetes-count');
  const rc = document.getElementById('high-count');

  if (tc) tc.textContent = history.length;
  if (hc) hc.textContent = heartCount;
  if (dc) dc.textContent = diabetesCount;
  if (rc) rc.textContent = highCount;
}

function renderHistory() {
  const history = getHistory();
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;

  const typeFilter = document.getElementById('filter-type')?.value || 'all';
  const riskFilter = document.getElementById('filter-risk')?.value || 'all';

  const filtered = history.filter(h => {
    if (typeFilter !== 'all' && h.type !== typeFilter) return false;
    if (riskFilter !== 'all' && h.risk_level !== riskFilter) return false;
    return true;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-state"><div class="empty-icon">📭</div><p>No records found. <a href="heart.html">Start an assessment</a></p></td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map((h, i) => `
    <tr>
      <td>${i + 1}</td>
      <td><span class="type-badge ${h.type}">${h.type === 'heart' ? '❤️ Heart' : '🩸 Diabetes'}</span></td>
      <td>${h.timestamp}</td>
      <td><span class="risk-pill ${h.risk_level === 'Low Risk' ? 'low' : h.risk_level === 'Medium Risk' ? 'medium' : 'high'}">${h.risk_level}</span></td>
      <td><strong>${h.probability.toFixed(2)}</strong></td>
      <td style="max-width:200px;font-size:0.82rem;color:var(--gray-600)">${h.recommendation}</td>
      <td><button class="dl-btn" onclick="downloadSingleReport(${h.id})">📄 Download</button></td>
    </tr>
  `).join('');

  updateSummary();
}

function clearHistory() {
  if (!confirm('Are you sure you want to clear all history? This cannot be undone.')) return;
  localStorage.removeItem('cdss_history');
  renderHistory();
  updateSummary();
}

// ============================================================
//  DOWNLOAD FUNCTIONS
// ============================================================

function downloadReport(type) {
  const last = type === 'heart' ? window._lastHeartResult : window._lastDiabetesResult;
  if (!last) return alert('No result to download yet.');
  generateReportHTML(type, last.data, last.inputs, last.data.risk_level);
}

function downloadSingleReport(id) {
  const history = getHistory();
  const entry = history.find(h => h.id === id);
  if (!entry) return;
  generateReportHTML(entry.type, entry, entry.inputs, entry.risk_level);
}

function generateReportHTML(type, data, inputs, risk) {
  const advice = ADVICE[type][risk];
  const typeLabel = type === 'heart' ? 'Heart Disease' : 'Diabetes';
  const icon = type === 'heart' ? '❤️' : '🩸';

  const inputRows = Object.entries(inputs || {}).map(([k, v]) =>
    `<tr><td style="padding:6px 12px;color:#475569;font-size:13px">${k}</td><td style="padding:6px 12px;font-weight:600;font-size:13px">${v}</td></tr>`
  ).join('');

  const riskColor = risk === 'High Risk' ? '#dc2626' : risk === 'Medium Risk' ? '#ca8a04' : '#16a34a';
  const riskBg = risk === 'High Risk' ? '#fee2e2' : risk === 'Medium Risk' ? '#fef9c3' : '#dcfce7';

  const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>CDSS Report — ${typeLabel}</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px; color: #1e293b; background: #f8fafc; }
    .report { max-width: 700px; margin: 0 auto; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.1); }
    .header { background: linear-gradient(135deg, #0a1628, #1354a3); color: white; padding: 36px 40px; }
    .header h1 { font-size: 22px; margin: 0 0 4px; }
    .header p { margin: 0; opacity: 0.65; font-size: 13px; }
    .body { padding: 32px 40px; }
    .risk-box { background: ${riskBg}; border: 2px solid ${riskColor}; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px; display: flex; align-items: center; gap: 20px; }
    .risk-label { font-size: 28px; font-weight: 700; color: ${riskColor}; }
    .risk-prob { font-size: 14px; color: #475569; margin-top: 4px; }
    .rec { font-size: 14px; color: #1e293b; background: #f1f5f9; border-radius: 8px; padding: 12px 16px; margin-bottom: 24px; }
    h2 { font-size: 15px; color: #0a1628; margin: 24px 0 12px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
    tr:nth-child(even) { background: #f8fafc; }
    ul { margin: 0; padding-left: 20px; }
    ul li { font-size: 13px; color: #475569; margin-bottom: 5px; line-height: 1.5; }
    .footer { text-align: center; font-size: 11px; color: #94a3b8; padding: 20px; border-top: 1px solid #e2e8f0; margin-top: 20px; }
  </style>
</head>
<body>
  <div class="report">
    <div class="header">
      <h1>${icon} CDSS — ${typeLabel} Risk Report</h1>
      <p>Generated: ${new Date().toLocaleString()} &nbsp;|&nbsp; Clinical Decision Support System</p>
    </div>
    <div class="body">
      <div class="risk-box">
        <div>
          <div class="risk-label">${risk}</div>
          <div class="risk-prob">Probability: ${typeof data.probability === 'number' ? data.probability.toFixed(2) : data.probability}</div>
        </div>
      </div>
      <div class="rec">📋 Recommendation: ${data.recommendation}</div>

      <h2>Patient Input Parameters</h2>
      <table>${inputRows}</table>

      <h2>🥗 Diet Recommendations</h2>
      <ul>${advice.diet.map(d => `<li>${d}</li>`).join('')}</ul>

      <h2>🏃 Exercise Recommendations</h2>
      <ul>${advice.exercise.map(e => `<li>${e}</li>`).join('')}</ul>

      <div class="footer">This report is for informational purposes only. Always consult a qualified physician for medical advice.<br/>✚ CDSS — Clinical Decision Support System</div>
    </div>
  </div>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `CDSS_${typeLabel.replace(' ', '_')}_Report_${Date.now()}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadAllCSV() {
  const history = getHistory();
  if (history.length === 0) return alert('No history to download.');

  const headers = ['#', 'Type', 'DateTime', 'RiskLevel', 'Probability', 'Prediction', 'Recommendation'];
  const rows = history.map((h, i) => [
    i + 1, h.type, h.timestamp, h.risk_level, h.probability, h.prediction, `"${h.recommendation}"`
  ]);

  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `CDSS_History_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================
//  UTILITIES
// ============================================================

function clearForm(type) {
  const inputs = document.querySelectorAll('input, select');
  inputs.forEach(el => el.value = '');
  if (type === 'heart') {
    document.getElementById('heart-result').style.display = 'none';
    hideError('heart-error');
  } else {
    document.getElementById('diabetes-result').style.display = 'none';
    hideError('diabetes-error');
  }
}

function showLoading() {
  const el = document.getElementById('loading');
  if (el) el.style.display = 'flex';
}

function hideLoading() {
  const el = document.getElementById('loading');
  if (el) el.style.display = 'none';
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.style.display = 'block'; }
}

function hideError(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}
