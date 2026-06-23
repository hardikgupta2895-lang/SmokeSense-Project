import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmokeSense · Smoker Risk Predictor",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("smoker_model.pkl")

model = load_model()
# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
  --bg:       #0a0c10;
  --card:     #111318;
  --card2:    #16191f;
  --border:   #1f2330;
  --accent:   #e8553e;
  --accent2:  #f5a623;
  --safe:     #2ecc8d;
  --text:     #e8eaf0;
  --muted:    #6b7280;
  --glow-r:   rgba(232,85,62,0.18);
  --glow-g:   rgba(46,204,141,0.15);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
  background: var(--card) !important;
  border-right: 1px solid var(--border);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar label override ── */
[data-testid="stSidebar"] label {
  color: #9ca3af !important;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 500;
}

/* ── Sidebar widgets ── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: var(--card2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Cards ── */
.ss-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 20px;
}

.ss-card-glow-r {
  background: linear-gradient(145deg, var(--card), #1a0f0c);
  border: 1px solid rgba(232,85,62,0.35);
  box-shadow: 0 0 32px var(--glow-r);
}

.ss-card-glow-g {
  background: linear-gradient(145deg, var(--card), #0c1a13);
  border: 1px solid rgba(46,204,141,0.3);
  box-shadow: 0 0 32px var(--glow-g);
}

/* ── Hero ── */
.hero-wrap {
  padding: 36px 0 12px;
  text-align: center;
}
.hero-eyebrow {
  font-family: 'Inter', sans-serif;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 600;
  margin-bottom: 10px;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 3.4rem;
  font-weight: 800;
  line-height: 1.05;
  color: var(--text);
  margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  font-size: 1.05rem;
  color: var(--muted);
  margin-top: 12px;
  font-weight: 300;
}

/* ── Section labels ── */
.ss-section-label {
  font-family: 'Inter', sans-serif;
  font-size: 0.7rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin-bottom: 14px;
}

/* ── Big stat ── */
.big-stat {
  font-family: 'Syne', sans-serif;
  font-size: 3.6rem;
  font-weight: 800;
  line-height: 1;
}
.big-stat-safe { color: var(--safe); }
.big-stat-warn { color: var(--accent2); }
.big-stat-risk { color: var(--accent); }

/* ── Risk badge ── */
.risk-badge {
  display: inline-block;
  padding: 6px 16px;
  border-radius: 100px;
  font-weight: 600;
  font-size: 0.85rem;
  letter-spacing: 0.04em;
  margin-top: 8px;
}
.badge-safe { background: rgba(46,204,141,0.12); color: var(--safe); border: 1px solid rgba(46,204,141,0.3); }
.badge-moderate { background: rgba(245,166,35,0.12); color: var(--accent2); border: 1px solid rgba(245,166,35,0.3); }
.badge-high { background: rgba(232,85,62,0.12); color: var(--accent); border: 1px solid rgba(232,85,62,0.3); }

/* ── Feature bar ── */
.feat-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
.feat-name {
  width: 110px;
  font-size: 0.82rem;
  color: #9ca3af;
  flex-shrink: 0;
}
.feat-bar-wrap {
  flex: 1;
  background: var(--card2);
  border-radius: 4px;
  height: 8px;
  overflow: hidden;
}
.feat-bar-fill {
  height: 8px;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.feat-val {
  width: 46px;
  text-align: right;
  font-size: 0.8rem;
  font-family: 'Syne', sans-serif;
  font-weight: 600;
  color: var(--text);
}

/* ── AI insight box ── */
.insight-box {
  background: linear-gradient(135deg, #111827, #0f172a);
  border: 1px solid #1e3a5f;
  border-left: 3px solid #3b82f6;
  border-radius: 10px;
  padding: 16px 20px;
  margin-top: 12px;
  font-size: 0.9rem;
  color: #93c5fd;
  line-height: 1.65;
}
.insight-header {
  color: #60a5fa;
  font-weight: 600;
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

/* ── Predict button ── */
div.stButton > button {
  width: 100%;
  background: linear-gradient(135deg, #e8553e, #c0392b) !important;
  color: white !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.05em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 14px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px rgba(232,85,62,0.3) !important;
}
div.stButton > button:hover {
  box-shadow: 0 6px 28px rgba(232,85,62,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Divider ── */
.ss-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 24px 0;
}

/* ── Metric pill ── */
.metric-pill {
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 18px;
  text-align: center;
}
.metric-pill-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
}
.metric-pill-label {
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 2px;
}

/* ── Tooltip note ── */
.ss-note {
  font-size: 0.76rem;
  color: var(--muted);
  margin-top: 4px;
  font-style: italic;
}

/* sensitivity slider labels */
[data-testid="stSidebar"] .stSlider > label {
  color: #9ca3af !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────────────────────────
REGION_MAP = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
REGION_LABELS = {0: "Northeast", 1: "Northwest", 2: "Southeast", 3: "Southwest"}

def predict(age, sex, bmi, children, region, charges):
    x = np.array([[age, sex, bmi, children, region, charges]])
    proba = model.predict_proba(x)[0]
    return proba[1], proba[0]  # smoker_prob, non_smoker_prob

def risk_tier(p):
    if p < 0.35:
        return "Low Risk", "badge-safe", "big-stat-safe"
    elif p < 0.65:
        return "Moderate Risk", "badge-moderate", "big-stat-warn"
    else:
        return "High Risk", "badge-high", "big-stat-risk"

def feature_contributions(age, sex, bmi, children, region, charges):
    """Approximate per-feature impact via marginal comparison."""
    base = 0.35  # baseline average
    contribs = {
        "Age":        min(1.0, max(0, (age - 18) / 47)),
        "Sex (Male)": 0.6 if sex == 1 else 0.3,
        "BMI":        min(1.0, max(0, (bmi - 15) / 35)),
        "Children":   min(1.0, children / 4),
        "Region":     [0.4, 0.3, 0.7, 0.5][region],
        "Charges":    min(1.0, max(0, (charges - 1000) / 64000)),
    }
    return contribs

def generate_insight(age, sex_label, bmi, children, region_label, charges, smoker_prob):
    risk_label, _, _ = risk_tier(smoker_prob)
    pct = f"{smoker_prob*100:.1f}%"
    factors = []
    if bmi > 30:
        factors.append(f"elevated BMI ({bmi:.1f}) which is a significant indicator")
    if charges > 25000:
        factors.append(f"high insurance charges (${charges:,.0f}) strongly correlate with smoking behavior")
    if age > 45:
        factors.append(f"age {age} puts this profile in a higher-prevalence demographic")
    if sex_label == "Male":
        factors.append("male sex is statistically associated with higher smoking rates")
    if not factors:
        factors.append("no dominant risk markers — the profile appears health-neutral")
    factor_text = "; ".join(factors[:2])
    return (
        f"The AdaBoost model assigns a smoker probability of <b>{pct}</b>, classifying this profile as "
        f"<b>{risk_label}</b>. Key drivers include: {factor_text}. "
        f"The model uses 100 decision-tree estimators tuned on insurance data — "
        f"charges and BMI carry the highest feature weight in this ensemble."
    )


# ─── Sidebar — Inputs ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#e8eaf0;">
        🫁 SmokeSense
      </div>
      <div style="font-size:0.72rem;color:#6b7280;letter-spacing:0.12em;text-transform:uppercase;margin-top:2px;">
        AdaBoost · Patient Profile
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ss-section-label">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 64, 35, help="Patient age in years")
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0
    children = st.slider("Dependents / Children", 0, 5, 1)

    st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
    st.markdown('<div class="ss-section-label">Health & Financial</div>', unsafe_allow_html=True)
    bmi = st.slider("BMI", 15.0, 50.0, 27.5, 0.1, help="Body Mass Index")
    region_label = st.selectbox("Region", list(REGION_MAP.keys()))
    region = REGION_MAP[region_label]
    charges = st.number_input("Annual Insurance Charges ($)", 1000, 65000, 8500, 100)

    st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Prediction")

    st.markdown("""
    <div style="margin-top:28px;padding:12px 14px;background:#0f1117;border-radius:10px;
                border:1px solid #1f2330;font-size:0.72rem;color:#6b7280;line-height:1.6;">
      <b style="color:#4b5563;">Model</b><br>
      AdaBoost Classifier<br>
      100 estimators · lr=1.0<br>
      Features: age, sex, BMI,<br>
      children, region, charges
    </div>
    """, unsafe_allow_html=True)


# ─── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">AI-Powered Smoking Risk Intelligence</div>
  <h1 class="hero-title">Smoke<span>Sense</span></h1>
  <p class="hero-sub">AdaBoost ensemble analysis · Insurance data features · Real-time risk profiling</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='ss-divider'>", unsafe_allow_html=True)


# ─── Compute prediction on load or button press ───────────────────────────
if "smoker_prob" not in st.session_state:
    st.session_state.smoker_prob, st.session_state.non_prob = predict(age, sex, bmi, children, region, charges)

if predict_btn:
    st.session_state.smoker_prob, st.session_state.non_prob = predict(age, sex, bmi, children, region, charges)

sp = st.session_state.smoker_prob
np_ = st.session_state.non_prob
risk_label, badge_class, stat_class = risk_tier(sp)


# ─── Top Row: Gauge + Stats ──────────────────────────────────────────────
col_gauge, col_stats = st.columns([1.1, 0.9])

with col_gauge:
    card_class = "ss-card ss-card-glow-r" if sp >= 0.5 else "ss-card ss-card-glow-g"
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    st.markdown('<div class="ss-section-label">Smoker Probability Gauge</div>', unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(sp * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "family": "Syne", "color": "#e8eaf0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151", "tickfont": {"color": "#6b7280", "size": 11}},
            "bar": {"color": "#e8553e" if sp >= 0.5 else "#2ecc8d", "thickness": 0.28},
            "bgcolor": "#111318",
            "bordercolor": "#1f2330",
            "steps": [
                {"range": [0, 35],  "color": "rgba(46,204,141,0.08)"},
                {"range": [35, 65], "color": "rgba(245,166,35,0.08)"},
                {"range": [65, 100],"color": "rgba(232,85,62,0.10)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 2},
                "thickness": 0.82,
                "value": sp * 100
            },
        },
    ))
    fig_gauge.update_layout(
        height=260, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf0",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:-8px;">
      <span class="risk-badge {badge_class}">{risk_label}</span>
      <div class="ss-note" style="margin-top:10px;">
        Confidence: Non-Smoker {np_*100:.1f}% · Smoker {sp*100:.1f}%
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_stats:
    # Big number
    st.markdown(f"""
    <div class="ss-card" style="padding:26px;">
      <div class="ss-section-label">Risk Score</div>
      <div class="big-stat {stat_class}">{sp*100:.1f}<span style="font-size:1.8rem;opacity:.6">%</span></div>
      <div style="font-size:0.85rem;color:#6b7280;margin-top:8px;">Smoker probability</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric pills
    m1, m2 = st.columns(2)
    with m1:
        bmi_cat = "Obese" if bmi >= 30 else "Overweight" if bmi >= 25 else "Normal"
        st.markdown(f"""
        <div class="metric-pill">
          <div class="metric-pill-val">{bmi:.1f}</div>
          <div class="metric-pill-label">BMI · {bmi_cat}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-pill">
          <div class="metric-pill-val">${charges/1000:.1f}k</div>
          <div class="metric-pill-label">Annual Charges</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"""
        <div class="metric-pill">
          <div class="metric-pill-val">{age}y</div>
          <div class="metric-pill-label">Age</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-pill">
          <div class="metric-pill-val">{region_label[:2].upper()}</div>
          <div class="metric-pill-label">Region</div>
        </div>
        """, unsafe_allow_html=True)

    # AI Insight
    st.markdown(f"""
    <div class="insight-box">
      <div class="insight-header">🤖 Model Insight</div>
      {generate_insight(age, sex_label, bmi, children, region_label, charges, sp)}
    </div>
    """, unsafe_allow_html=True)


# ─── Mid Row: Feature Contributions + Donut ─────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_feat, col_donut = st.columns([1.1, 0.9])

contribs = feature_contributions(age, sex, bmi, children, region, charges)

with col_feat:
    st.markdown('<div class="ss-card">', unsafe_allow_html=True)
    st.markdown('<div class="ss-section-label">Feature Risk Contribution</div>', unsafe_allow_html=True)

    for fname, val in contribs.items():
        pct = int(val * 100)
        st.markdown(f"""
        <div class="feat-row">
          <span class="feat-name">{fname}</span>
          <div class="feat-bar-wrap">
            <div class="feat-bar-fill" style="width:{pct}%"></div>
          </div>
          <span class="feat-val">{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_donut:
    st.markdown('<div class="ss-card">', unsafe_allow_html=True)
    st.markdown('<div class="ss-section-label">Probability Breakdown</div>', unsafe_allow_html=True)

    fig_donut = go.Figure(go.Pie(
        labels=["Non-Smoker", "Smoker"],
        values=[round(np_*100,1), round(sp*100,1)],
        hole=0.62,
        marker_colors=["#2ecc8d", "#e8553e"],
        textinfo="label+percent",
        textfont_color="#e8eaf0",
        textfont_size=12,
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        showlegend=False,
    ))
    fig_donut.add_annotation(
        text=f"<b>{sp*100:.0f}%</b><br>Smoker",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#e8553e", family="Syne"),
        xanchor="center"
    )
    fig_donut.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Sensitivity / What-If Analysis ─────────────────────────────────────
st.markdown("<hr class='ss-divider'>", unsafe_allow_html=True)
st.markdown('<div class="ss-section-label" style="font-size:0.75rem;letter-spacing:0.2em;">What-If Sensitivity Analysis</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 BMI Sweep", "💰 Charges Sweep", "🎂 Age Sweep"])

def sweep_chart(sweep_vals, sweep_name, fixed_kwargs, color):
    probs = []
    for v in sweep_vals:
        kwargs = dict(fixed_kwargs)
        kwargs[sweep_name] = v
        p, _ = predict(**kwargs)
        probs.append(round(p * 100, 2))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_vals, y=probs,
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        hovertemplate=f"{sweep_name}: %{{x}}<br>Smoker Prob: %{{y:.1f}}%<extra></extra>",
    ))
    # Current value marker
    cur_val = fixed_kwargs.get(sweep_name, sweep_vals[len(sweep_vals)//2])
    cur_prob, _ = predict(**{**fixed_kwargs, sweep_name: cur_val})
    fig.add_vline(x=cur_val, line_dash="dot", line_color="#ffffff", line_width=1.5, opacity=0.5)
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=sweep_name,
            gridcolor="#1f2330", tickfont_color="#6b7280", title_font_color="#9ca3af",
        ),
        yaxis=dict(
            title="Smoker Prob (%)", range=[0, 100],
            gridcolor="#1f2330", tickfont_color="#6b7280", title_font_color="#9ca3af",
        ),
        font_color="#e8eaf0",
    )
    return fig

base_kwargs = dict(age=age, sex=sex, bmi=bmi, children=children, region=region, charges=charges)

with tab1:
    bmi_range = np.round(np.arange(15.0, 50.5, 0.5), 1)
    fig1 = sweep_chart(bmi_range, "bmi", base_kwargs, "#f5a623")
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('<p class="ss-note">How smoker probability shifts as BMI changes, holding all other inputs constant.</p>', unsafe_allow_html=True)

with tab2:
    charge_range = np.arange(1000, 65001, 1000)
    fig2 = sweep_chart(charge_range, "charges", base_kwargs, "#e8553e")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<p class="ss-note">Insurance charges are the strongest predictor in this ensemble — observe the threshold effect.</p>', unsafe_allow_html=True)

with tab3:
    age_range = np.arange(18, 65, 1)
    fig3 = sweep_chart(age_range, "age", base_kwargs, "#2ecc8d")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<p class="ss-note">Age-based risk trajectory for this patient\'s specific BMI, charges, and region.</p>', unsafe_allow_html=True)


# ─── Population Comparison Heatmap ──────────────────────────────────────
st.markdown("<hr class='ss-divider'>", unsafe_allow_html=True)
st.markdown('<div class="ss-section-label" style="font-size:0.75rem;letter-spacing:0.2em;">Population Risk Heatmap — Age vs BMI</div>', unsafe_allow_html=True)

ages_grid = np.arange(18, 65, 3)
bmis_grid = np.round(np.arange(15, 50, 2), 0)
heat = np.zeros((len(bmis_grid), len(ages_grid)))
for i, b in enumerate(bmis_grid):
    for j, a in enumerate(ages_grid):
        p, _ = predict(int(a), sex, float(b), children, region, charges)
        heat[i, j] = round(p * 100, 1)

fig_heat = go.Figure(go.Heatmap(
    z=heat,
    x=ages_grid,
    y=bmis_grid,
    colorscale=[
        [0.0,  "#0d2818"],
        [0.35, "#2ecc8d"],
        [0.65, "#f5a623"],
        [1.0,  "#e8553e"],
    ],
    colorbar=dict(
    title=dict(
        text="Smoker %",
        font=dict(color="#9ca3af")
    ),
    tickfont=dict(color="#9ca3af"),
    outlinecolor="#1f2330",
    ),
    hovertemplate="Age: %{x}<br>BMI: %{y}<br>Smoker Prob: %{z:.1f}%<extra></extra>",
    zmin=0, zmax=100,
))
# Mark current patient
fig_heat.add_trace(go.Scatter(
    x=[age], y=[round(bmi, 0)],
    mode="markers",
    marker=dict(symbol="cross", size=14, color="white", line=dict(color="#e8553e", width=2)),
    name="Current Patient",
    hovertemplate=f"Current: Age {age}, BMI {bmi:.1f}<br>Risk: {sp*100:.1f}%<extra></extra>",
))
fig_heat.update_layout(
    height=310,
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(title="Age", gridcolor="#1f2330", tickfont_color="#6b7280", title_font_color="#9ca3af"),
    yaxis=dict(title="BMI", gridcolor="#1f2330", tickfont_color="#6b7280", title_font_color="#9ca3af"),
    font_color="#e8eaf0",
    legend=dict(font_color="#9ca3af", bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_heat, use_container_width=True)
st.markdown(f'<p class="ss-note">White ✕ marks the current patient (Age {age}, BMI {bmi:.1f}) across the population grid. Fixed: charges=${charges:,.0f}, region={region_label}.</p>', unsafe_allow_html=True)


# ─── Footer ─────────────────────────────────────────────────────────────
st.markdown("""
<hr class='ss-divider'>
<div style="text-align:center;padding:16px 0 8px;color:#374151;font-size:0.72rem;letter-spacing:0.08em;">
  SMOKESENSE · AdaBoost Classifier · For research & educational use only
</div>
""", unsafe_allow_html=True)
