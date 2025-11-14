# streamlit_sports_app.py
"""
Streamlit app — Manual entry + analysis for Hockey and Football athletes.

Run:
    pip install streamlit pandas numpy matplotlib
    streamlit run streamlit_sports_app.py
"""

from typing import Dict, Any
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import base64
import datetime

st.set_page_config(page_title="Sports Performance Analyzer", layout="wide")

# --------------- Utility functions ---------------

def compute_acwr(acute: float, chronic: float) -> float:
    try:
        if chronic <= 0:
            return float('inf') if acute > 0 else 1.0
        return acute / chronic
    except Exception:
        return np.nan

def normalize(value, min_v, max_v):
    if pd.isna(value):
        return 0.0
    return float(np.clip((value - min_v) / max(1e-6, (max_v - min_v)), 0, 1))

def weighted_score(weights: Dict[str,float], vals: Dict[str,float]) -> float:
    total_w = sum(weights.values())
    if total_w == 0:
        return 0.0
    s = 0.0
    for k,w in weights.items():
        v = vals.get(k, 0.0)
        s += w * v
    return 100.0 * (s / total_w)

def estimate_injury_risk(acwr, hrv, sleep_hours, prev_injury, age):
    """
    Simple heuristic injury risk between 0 and 1:
    - ACWR >> 1.5 increases risk
    - Low HRV increases risk
    - Low sleep increases risk
    - Previous injury and older age increase risk
    """
    score = 0.0
    # ACWR contribution
    if np.isfinite(acwr):
        if acwr >= 1.8:
            score += 0.35
        elif acwr >= 1.4:
            score += 0.20
        elif acwr >= 1.2:
            score += 0.08
        else:
            score += 0.00
    # HRV (lower HRV -> higher risk) assume HRV typical 20-60 ms
    if not pd.isna(hrv):
        if hrv < 20:
            score += 0.20
        elif hrv < 30:
            score += 0.12
        elif hrv < 40:
            score += 0.06
    # Sleep
    if not pd.isna(sleep_hours):
        if sleep_hours < 5:
            score += 0.18
        elif sleep_hours < 7:
            score += 0.08
    # previous injury
    if prev_injury:
        score += 0.12
    # age effect (per 5 years above 25 add small risk)
    if not pd.isna(age):
        age = float(age)
        if age > 25:
            score += min(0.12, 0.02 * ((age - 25)//5 + 1))
    # clamp
    return float(np.clip(score, 0.0, 0.98))

def make_text_summary(sport, inputs, perf, inj_prob, notes):
    lines = []
    lines.append(f"Sport: {sport}")
    lines.append(f"Session date: {inputs.get('timestamp', '')}")
    lines.append("Inputs:")
    for k,v in inputs.items():
        if k in ('timestamp',):
            continue
        lines.append(f"  - {k}: {v}")
    lines.append(f"\nPredicted performance index: {perf:.2f} / 100")
    lines.append(f"Estimated injury probability (next window): {inj_prob:.2%}")
    lines.append("\nRecommendations:")
    for n in notes:
        lines.append(" - " + n)
    return "\n".join(lines)

def download_text(text: str, filename: str = "analysis.txt"):
    b = base64.b64encode(text.encode()).decode()
    href = f"data:text/plain;base64,{b}"
    return href

# --------------- Sport-specific definitions ---------------

SPORT_FEATURES = {
    "Hockey": {
        # feature_name: (label, min, max, description)
        "total_distance": ("Total distance (m)", 0, 15000, "Session distance covered in meters"),
        "max_speed": ("Max speed (m/s)", 0, 12, "Peak running speed"),
        "sprint_count": ("Sprint count", 0, 50, "Number of sprints"),
        "avg_hr": ("Average heart rate (bpm)", 40, 210, "Average heart rate"),
        "hrv": ("HRV (ms)", 5, 100, "Heart rate variability"),
        "shots_on_target": ("Shots on target", 0, 20, "Shots on target / attempts"),
        "passes_completed": ("Passes completed", 0, 200, "Successful passes"),
        "acute_load": ("Acute load", 0, 2000, "7-day load"),
        "chronic_load": ("Chronic load", 0, 2000, "28-day load"),
        "sleep_hours": ("Sleep (hours)", 0, 12, "Previous night's sleep"),
        "age": ("Age (years)", 14, 50, "Athlete age")
    },
    "Football": {
        "total_distance": ("Total distance (m)", 0, 16000, "Session distance covered in meters"),
        "max_speed": ("Max speed (m/s)", 0, 12.5, "Peak running speed"),
        "sprint_count": ("Sprint count", 0, 60, "Number of sprints"),
        "avg_hr": ("Average heart rate (bpm)", 40, 210, "Average heart rate"),
        "hrv": ("HRV (ms)", 5, 100, "Heart rate variability"),
        "goals": ("Goals", 0, 10, "Goals scored this session"),
        "successful_tackles": ("Successful tackles", 0, 50, "Defensive actions"),
        "acute_load": ("Acute load", 0, 2500, "7-day load"),
        "chronic_load": ("Chronic load", 0, 2500, "28-day load"),
        "sleep_hours": ("Sleep (hours)", 0, 12, "Previous night's sleep"),
        "age": ("Age (years)", 14, 50, "Athlete age")
    }
}

# Weights for subdomains for performance index (normalized later)
# For each sport, map features to normalized 0..1 components and give them domain weights.
SPORT_WEIGHTS = {
    "Hockey": {
        "movement": {"total_distance": 0.25, "max_speed": 0.25, "sprint_count": 0.2},
        "physio": {"avg_hr": 0.15, "hrv": 0.1},
        "skill": {"shots_on_target": 0.3, "passes_completed": 0.2}
    },
    "Football": {
        "movement": {"total_distance": 0.25, "max_speed": 0.25, "sprint_count": 0.2},
        "physio": {"avg_hr": 0.15, "hrv": 0.1},
        "skill": {"goals": 0.35, "successful_tackles": 0.25}
    }
}

# --------------- App UI ---------------

st.title("Sports Performance Analyzer")
st.markdown("Choose a sport, enter session data, and get immediate analysis (performance index, injury risk, and recommendations).")

with st.sidebar:
    st.markdown("## Controls")
    sport = st.selectbox("Select sport", ["Hockey", "Football"])
    st.markdown("---")
    use_demo = st.checkbox("Use demo default values for quick test", value=False)
    st.markdown("App date: " + datetime.date.today().isoformat())

# Build form inputs dynamically
st.header(f"Enter session data — {sport}")

features = SPORT_FEATURES[sport]

# Two-column layout for inputs
cols = st.columns(2)
input_vals = {}
i = 0
for fname, meta in features.items():
    label, min_v, max_v, desc = meta
    col = cols[i % 2]
    if fname in ("age",):
        input_vals[fname] = col.number_input(label, min_value=int(min_v), max_value=int(max_v), value=int(24) if use_demo else int(24))
    elif fname in ("sleep_hours",):
        input_vals[fname] = col.slider(label, float(min_v), float(max_v), float(7.5) if use_demo else float(7.5), step=0.25)
    elif fname in ("avg_hr", "hrv", "max_speed"):
        # allow float input
        input_vals[fname] = col.number_input(label, min_value=float(min_v), max_value=float(max_v), value=float((min_v+max_v)/4) if use_demo else float((min_v+max_v)/4), format="%.2f")
    else:
        # integer-like
        default_val = 0
        if use_demo:
            if fname in ("total_distance",):
                default_val = 6000 if sport=="Hockey" else 7500
            elif fname in ("sprint_count",):
                default_val = 8
            elif fname in ("shots_on_target", "goals"):
                default_val = 2
            elif fname in ("passes_completed", "successful_tackles"):
                default_val = 40
            elif fname in ("acute_load",):
                default_val = 460
            elif fname in ("chronic_load",):
                default_val = 500
        input_vals[fname] = col.number_input(label, min_value=int(min_v), max_value=int(max_v), value=int(default_val))

# extra context
st.subheader("Context & history")
col1, col2 = st.columns(2)
athlete_id = col1.text_input("Athlete ID", value="A01")
prev_injury = col2.checkbox("Previous recent injury (within past 6 months)?", value=False)
session_date = col2.date_input("Session date", value=datetime.date.today())

# Run analysis button
if st.button("Run analysis"):
    # prepare normalized component values (0..1)
    comps = {}
    sport_weights = SPORT_WEIGHTS[sport]

    # Movement components
    movement_vals = {}
    for f in sport_weights["movement"].keys():
        mn, mx = features[f][1], features[f][2]
        movement_vals[f] = normalize(input_vals.get(f, np.nan), mn, mx)
    # Physio components
    physio_vals = {}
    for f in sport_weights["physio"].keys():
        mn, mx = features[f][1], features[f][2]
        physio_vals[f] = normalize(input_vals.get(f, np.nan), mn, mx)
    # Skill components
    skill_vals = {}
    for f in sport_weights["skill"].keys():
        mn, mx = features[f][1], features[f][2]
        skill_vals[f] = normalize(input_vals.get(f, np.nan), mn, mx)

    # compute domain scores as weighted average of normalized features inside domain
    def domain_score(mapping, vals):
        if not mapping:
            return 0.0
        s = 0.0
        wsum = 0.0
        for k,w in mapping.items():
            s += w * vals.get(k, 0.0)
            wsum += w
        return s / max(1e-6, wsum)

    movement_score = domain_score(sport_weights["movement"], movement_vals)
    physio_score = domain_score(sport_weights["physio"], physio_vals)
    skill_score = domain_score(sport_weights["skill"], skill_vals)

    # overall performance: combine domain scores, weigh skill slightly higher
    overall_weights = {"movement": 0.33, "physio": 0.17, "skill": 0.5}
    perf_index = 100.0 * (overall_weights["movement"]*movement_score + overall_weights["physio"]*physio_score + overall_weights["skill"]*skill_score)

    # injury risk
    acute = float(input_vals.get("acute_load", np.nan))
    chronic = float(input_vals.get("chronic_load", np.nan))
    acwr = compute_acwr(acute, chronic)
    hrv = float(input_vals.get("hrv", np.nan))
    sleep_h = float(input_vals.get("sleep_hours", np.nan))
    age = float(input_vals.get("age", np.nan))
    inj_prob = estimate_injury_risk(acwr, hrv, sleep_h, prev_injury, age)

    # recommendations
    notes = []
    # ACWR advice
    if np.isfinite(acwr):
        notes.append(f"ACWR = {acwr:.2f}")
        if acwr > 1.6:
            notes.append("ACWR high — reduce load by 15–25%, prioritize recovery sessions.")
        elif acwr > 1.3:
            notes.append("ACWR moderate — close monitoring recommended; reduce extremes.")
        else:
            notes.append("ACWR acceptable.")
    else:
        notes.append("ACWR not computable (chronic_load <= 0).")

    # HRV and sleep
    if hrv < 25:
        notes.append("Low HRV — athlete may be fatigued; monitor HRV daily and reduce high-intensity work.")
    if sleep_h < 6:
        notes.append("Low sleep — prioritize ≥7 hours sleep for recovery.")

    # injury probability messaging
    if inj_prob > 0.6:
        notes.append("High injury risk — medical review and rest suggested.")
    elif inj_prob > 0.3:
        notes.append("Moderate injury risk — reduce load and monitor symptoms.")
    else:
        notes.append("Low immediate injury risk — proceed with planned training.")

    # age and prev injury
    if prev_injury:
        notes.append("Previous injury — consider physiotherapy check and modified load.")

    # Present results
    st.subheader("Results")
    colp, coli = st.columns(2)
    colp.metric("Performance index", f"{perf_index:.1f} / 100")
    coli.metric("Estimated injury probability", f"{inj_prob:.1%}")

    # Radar-like bar chart for domain breakdown
    domains = ["Movement", "Physio", "Skill"]
    domain_vals = [movement_score*100, physio_score*100, skill_score*100]
    fig, ax = plt.subplots(figsize=(6,3))
    bars = ax.bar(domains, domain_vals, alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Domain score (0-100)")
    for bar, val in zip(bars, domain_vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.0f}", ha='center')
    st.pyplot(fig)

    # Quick text summary and recommendations
    st.subheader("Recommendations")
    for n in notes:
        st.write("- " + n)

    # Detailed summary and download
    inputs_for_summary = input_vals.copy()
    inputs_for_summary['athlete_id'] = athlete_id
    inputs_for_summary['timestamp'] = pd.to_datetime(session_date).isoformat()
    summary_text = make_text_summary(sport, inputs_for_summary, perf_index, inj_prob, notes)
    st.subheader("Detailed summary (text)")
    st.code(summary_text)

    st.markdown(f"[Download summary as text]({download_text(summary_text)})")
