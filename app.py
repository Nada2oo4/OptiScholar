"""
OptiScholar — Full App with Login System
==========================================
Features:
  - Signup / Login with SQLite persistence
  - Personalized dashboard per student
  - Applied / Saved / Ignored scholarship feedback
  - Deadline reminders and notifications
  - SHAP explanations per recommendation
  - Feedback → implicit ratings (ready for model retraining)
  - Groq LLM chatbot (free API)

Run:
    streamlit run app.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import streamlit as st

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR   = os.path.join(BASE, "models")
ROBERTA_DIR  = os.path.join(BASE, "fine_tuned_roberta")

SCHOLARSHIPS_PATH  = os.path.join(DATA_DIR, "scholarships_final_ready-2.csv")
STUDENTS_PATH      = os.path.join(DATA_DIR, "students.csv")
INTERACTIONS_PATH  = os.path.join(DATA_DIR, "interactions.csv")
TRANSFER_FN_PATH   = os.path.join(MODELS_DIR, "transfer_fn_v2.pt")
NCF_MODEL_PATH     = os.path.join(MODELS_DIR, "ncf_model_v2.pt")

sys.path.insert(0, BASE)

st.set_page_config(
    page_title="OptiScholar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import database ────────────────────────────────────────────
from database import Database
DB = Database(os.path.join(BASE, "optischolar.db"))


# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');
    :root {
        --primary:#1e3a8a; --primary-light:#2d54c5;
        --gold:#f59e0b; --surface:#f8f9fc; --border:#e5e7eb;
        --text:#1a1f2e; --muted:#6b7280;
        --green:#10b981; --red:#ef4444; --amber:#f59e0b;
    }
    html,body,[class*="css"] {
        font-family:'DM Sans',sans-serif; color:var(--text);
    }
    h1,h2,h3 {
        font-family:'DM Serif Display',serif !important;
        color:var(--primary) !important;
    }
    [data-testid="stSidebar"] { background:var(--primary) !important; }
    [data-testid="stSidebar"] * { color:white !important; }
    [data-testid="collapsedControl"] {
        display:block !important; visibility:visible !important;
        background:var(--primary) !important;
        border-radius:0 8px 8px 0 !important;
    }
    [data-testid="collapsedControl"] svg { fill:white !important; }
    .stButton > button {
        background:var(--primary) !important; color:white !important;
        border:none !important; border-radius:8px !important;
        font-weight:500 !important;
    }
    .stButton > button:hover { background:var(--primary-light) !important; }
    .stTextInput input {
        color:#1a1f2e !important; background:white !important;
    }
    /* chat_input cursor only — keep original styling */
    [data-testid="stChatInput"] textarea {
        caret-color:#1e3a8a !important;
    }
    .chat-user {
        background:var(--primary) !important; color:white !important;
        border-radius:18px 18px 4px 18px; padding:10px 16px;
        margin:6px 0; max-width:75%; float:right; clear:both;
    }
    .chat-bot {
        background:#f0f4ff !important; color:#1a1f2e !important;
        border:1px solid var(--border);
        border-radius:18px 18px 18px 4px; padding:10px 16px;
        margin:6px 0; max-width:80%; float:left; clear:both;
    }
    .chat-bot *,.chat-user * { color:inherit !important; }
    .action-applied { color:var(--green); font-weight:600; }
    .action-saved   { color:var(--amber); font-weight:600; }
    .action-ignored { color:var(--muted); font-weight:600; }
    #MainMenu,footer { visibility:hidden; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════

def init_session():
    for k, v in {
        "user":            None,   # logged-in user dict
        "page":            "🏠 Dashboard",
        "profile":         None,
        "recommendations": None,
        "gap_results":     None,
        "opto_df":         None,
        "chatbot":         None,
        "_models":         None,
        "_ner_pipeline":   None,
        "_last_X":         None,
        "_shap_vals":      None,
        "chat_history":    [],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
# DEVICE
# ══════════════════════════════════════════════════════════════

def get_device():
    import torch
    return torch.device("cpu")  # CPU for stability on M1


# ══════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_scholarships():
    import pandas as pd
    df = pd.read_csv(SCHOLARSHIPS_PATH)
    df.columns = df.columns.str.strip()
    for c in ["min_gpa_required","funding_amount_raw","requires_financial_need",
               "eligible_bachelor","eligible_master","eligible_phd","eligible_high_school"]:
        df[c] = pd.to_numeric(df.get(c,0), errors="coerce").fillna(0)
    for c in ["scholarship_title","scholarship_id","description_cleaned",
               "scholarship_type","citizenship_required","link","deadline_date"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    import torch, numpy as np, pandas as pd
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler

    DEVICE = get_device()
    interactions = pd.read_csv(INTERACTIONS_PATH)
    students     = pd.read_csv(STUDENTS_PATH)

    TYPES   = ["Merit-Based","Need-Based","Academic Excellence",
                "Community Service","Athletic"]
    le_type = LabelEncoder().fit(TYPES)
    scaler  = MinMaxScaler().fit(
        students[["final_gpa","household_income","age"]].fillna(0)
    )
    encoders = {
        "le_type":  le_type, "scaler": scaler, "DEVICE": DEVICE,
        "amt_max":  float(interactions["amount"].max()),
        "type_map": {t:i for i,t in enumerate(TYPES)},
    }

    student_embs = None
    try:
        from ncf_pipeline import NCFModel
        le_stu = LabelEncoder().fit(interactions["student_id"])
        le_sch = LabelEncoder().fit(interactions["scholarship_id"])
        ncf    = NCFModel(
            len(le_stu.classes_), len(le_sch.classes_),
            emb_dim=8, hidden=[32,16], n_sf=6, n_cf=2, dropout=0.3
        ).to(DEVICE)
        ncf.load_state_dict(torch.load(NCF_MODEL_PATH, map_location=DEVICE, weights_only=False))
        ncf.eval()
        with torch.no_grad():
            idx = torch.arange(len(le_stu.classes_), dtype=torch.long).to(DEVICE)
            student_embs = ncf.get_student_embedding(idx).cpu().numpy()
    except Exception as e:
        st.sidebar.warning(f"NCF: {e}")

    tf = None
    try:
        from ncf_pipeline import TransferFunction
        tf = TransferFunction(in_dim=21, hidden=[128,64,32], dropout=0.3).to(DEVICE)
        tf.load_state_dict(torch.load(TRANSFER_FN_PATH, map_location=DEVICE, weights_only=False))
        tf.eval()
    except Exception as e:
        st.sidebar.warning(f"Transfer Fn: {e}")

    return tf, student_embs, encoders


@st.cache_resource(show_spinner=False)
def load_ner():
    from transformers import pipeline
    model_path = ROBERTA_DIR if os.path.exists(ROBERTA_DIR) \
                 else "venkatasagar/NER-roBERTa-finetuned"
    return pipeline("ner", model=model_path,
                    aggregation_strategy="simple", device=-1)


# ══════════════════════════════════════════════════════════════
# DOCUMENT PARSER (RoBERTa NER + Regex)
# ══════════════════════════════════════════════════════════════

NER_LABEL_MAP = {
    "LABEL_38":"B-NAME","LABEL_93":"I-NAME",
    "LABEL_27":"B-GPA_KEY","LABEL_86":"B-GPA_VALUE","LABEL_11":"B-GPA_VALUE",
    "LABEL_18":"B-DEGREE","LABEL_76":"I-DEGREE",
    "LABEL_25":"B-FIELD","LABEL_83":"I-FIELD",
    "LABEL_40":"B-INSTITUTION","LABEL_94":"I-INSTITUTION",
    "LABEL_92":"B-LOCATION","LABEL_111":"O",
}

def parse_document(pdf_path: str) -> dict:
    import re, datetime

    # Extract text
    text = ""
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = "\n".join(p.get_text("text") for p in doc)
        doc.close()
    except Exception:
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            return {"success": False, "fields": {}}

    if not text.strip():
        return {"success": False, "fields": {}}

    fields = {}
    tl     = text.lower()

    # ── RoBERTa NER ────────────────────────────────────────────
    nlp = st.session_state.get("_ner_pipeline")
    if nlp:
        cutoff = min(1500,
            tl.find("assessment notation") if "assessment notation" in tl else 1500,
            tl.find("high distinction")    if "high distinction"    in tl else 1500,
        )
        excerpt = text[:max(cutoff, 400)]
        try:
            raw     = nlp(excerpt)
            grouped = {}
            for ent in raw:
                mapped = NER_LABEL_MAP.get(ent["entity_group"],"O")
                if mapped == "O" or ent["score"] < 0.5: continue
                grouped.setdefault(mapped,[]).append(
                    {"text": ent["word"].strip(), "score": float(ent["score"])}
                )

            # Name
            name_parts = []
            for lab in ["B-NAME","I-NAME"]:
                for c in grouped.get(lab,[]):
                    if not any(b in c["text"].lower() for b in
                               ["transcript","university","official","institute"]):
                        name_parts.append((c["text"],c["score"]))
            if name_parts and len(name_parts[0][0].split()) >= 1:
                name = " ".join(p[0] for p in name_parts)
                conf = sum(p[1] for p in name_parts)/len(name_parts)
                fields["person_name"] = {"value":name,"confidence":round(conf,3),"source":"roberta"}

            # GPA
            for c in grouped.get("B-GPA_VALUE",[]):
                raw_v = c["text"].strip(".:,")
                try:
                    gpa = float(raw_v.split("/")[0]) if "/" in raw_v else float(raw_v)
                    if gpa > 4: gpa = round(gpa/7*4,2)
                    if 0 <= gpa <= 4:
                        fields["final_gpa"] = {"value":round(gpa,2),"confidence":round(c["score"],3),"source":"roberta"}
                        break
                except ValueError: pass

            # Degree
            dp = [c for c in grouped.get("B-DEGREE",[])
                  if not any(f in c["text"].lower() for f in ["faculty","school","college"])]
            if dp:
                best = max(dp, key=lambda x:x["score"])
                dt   = " ".join([best["text"]]+[c["text"] for c in grouped.get("I-DEGREE",[])]).lower()
                lvl  = ("phd" if any(w in dt for w in ["phd","doctor"]) else
                         "master" if any(w in dt for w in ["master","msc","mba"]) else
                         "bachelor" if any(w in dt for w in ["bachelor","b.sc","b.tech"]) else None)
                if lvl:
                    fields["degree_level"] = {"value":lvl,"confidence":round(best["score"]*0.9,3),"source":"roberta"}

            # Field
            fp = [(c["text"],c["score"]) for lab in ["B-FIELD","I-FIELD"]
                  for c in grouped.get(lab,[])]
            if fp:
                fv = " ".join(p[0] for p in fp).title()
                if 3 <= len(fv) <= 80:
                    fields["field_of_study"] = {"value":fv,"confidence":round(sum(p[1] for p in fp)/len(fp),3),"source":"roberta"}

        except Exception:
            pass

    # ── Regex fallback ─────────────────────────────────────────
    # GPA
    if "final_gpa" not in fields:
        m = re.search(r'current\s+gpa[\s:]+(\d+\.?\d{1,2})', tl)
        if m:
            val = float(m.group(1))
            fields["final_gpa"] = {"value":round(val/7*4,2) if val>4 else round(val,2),"confidence":0.88,"source":"regex"}
    if "final_gpa" not in fields:
        for pat in [r'\bc?gpa[\s:=]+(\d+\.?\d{1,2})\b',r'cumulative\s+gpa[\s:]+(\d+\.?\d{1,2})']:
            m = re.search(pat, tl)
            if m:
                try:
                    gpa = float(m.group(1))
                    if 0 <= gpa <= 4:
                        fields["final_gpa"] = {"value":round(gpa,2),"confidence":0.88,"source":"regex"}
                        break
                except: pass
    if "final_gpa" not in fields:
        if any(kw in tl for kw in ["degree year","module","award programme"]):
            marks = [float(x) for x in re.findall(r'(?<!\d)([4-9]\d|100)(?!\d)',tl) if 40<=float(x)<=100]
            if len(marks)>=3:
                avg = sum(marks)/len(marks)
                gpa = 4.0 if avg>=70 else 3.3 if avg>=60 else 2.7 if avg>=50 else 2.0
                fields["final_gpa"] = {"value":round(gpa,2),"confidence":0.82,"source":"regex_marks"}

    # Degree level
    if "degree_level" not in fields:
        if re.search(r'degree\s+year\s+(one|two|three|four|\d)',tl):
            fields["degree_level"] = {"value":"bachelor","confidence":0.92,"source":"regex"}
        elif re.search(r'\b(ph\.?d|doctor(?:ate)?)\b',tl):
            fields["degree_level"] = {"value":"phd","confidence":0.95,"source":"regex"}
        elif re.search(r'\b(master|m\.?sc|mba)\b',tl):
            fields["degree_level"] = {"value":"master","confidence":0.88,"source":"regex"}
        elif re.search(r'\b(bachelor|b\.?sc|undergraduate)\b',tl):
            fields["degree_level"] = {"value":"bachelor","confidence":0.88,"source":"regex"}

    # Name
    if "person_name" not in fields:
        m = re.search(r'student\s+name\s*[:=]\s*([A-Za-z][A-Za-z\s]{3,60}?)(?:\n|birth|id)',
                      text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            fields["person_name"] = {"value":name,"confidence":0.92,"source":"regex"}

    # Age
    for pat in [r'(?:birth\s*date|date\s*of\s*birth|dob)[\s:]+(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})',
                r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})']:
        m = re.search(pat, tl)
        if m:
            try:
                d,mo,y = int(m.group(1)),int(m.group(2)),int(m.group(3))
                if 1985<=y<=2010:
                    import datetime as dt
                    dob   = dt.date(y,mo,d)
                    today = dt.date.today()
                    age   = today.year-dob.year-((today.month,today.day)<(dob.month,dob.day))
                    if 13<=age<=35:
                        fields["age"] = {"value":age,"confidence":0.88,"source":"regex"}
                        break
            except: pass

    # Field
    if "field_of_study" not in fields:
        m = re.search(r'award\s+programme\s*[:=]\s*([A-Za-z\s&,]+?)(?:\n|marking)',
                      text, re.IGNORECASE)
        if m:
            f = m.group(1).strip().title()
            if 3<=len(f)<=80:
                fields["field_of_study"] = {"value":f,"confidence":0.80,"source":"regex"}

    # Gender
    if re.search(r'\bgender[\s:]+male\b',tl):
        fields["gender"] = {"value":"Male","confidence":0.95,"source":"regex"}
    elif re.search(r'\bgender[\s:]+female\b',tl):
        fields["gender"] = {"value":"Female","confidence":0.95,"source":"regex"}

    return {"success":True,"fields":fields}


def prefill_from_parse(result):
    f        = result.get("fields",{})
    deg_opts = ["bachelor","master","phd","high_school"]
    deg_val  = f.get("degree_level",{}).get("value","bachelor")
    gen_opts = ["Male","Female"]
    gen_val  = f.get("gender",{}).get("value","Male")
    return {
        "final_gpa":        f.get("final_gpa",{}).get("value",2.5),
        "degree_level_idx": deg_opts.index(deg_val) if deg_val in deg_opts else 0,
        "age":              f.get("age",{}).get("value",20),
        "gender_idx":       gen_opts.index(gen_val) if gen_val in gen_opts else 0,
        "field_of_study":   f.get("field_of_study",{}).get("value",""),
        "person_name":      f.get("person_name",{}).get("value",""),
    }


# ══════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════

def run_recommendations(profile, opto_df, top_n=20):
    import torch, numpy as np

    tf, student_embs, encoders = st.session_state["_models"]
    if tf is None or student_embs is None:
        return None

    DEVICE   = encoders["DEVICE"]
    scaler   = encoders["scaler"]
    amt_max  = encoders["amt_max"]
    type_map = encoders["type_map"]

    gpa    = float(profile.get("final_gpa") or 2.5)
    income = float(profile.get("household_income",50000))
    age    = float(profile.get("age",20))
    cont   = scaler.transform([[gpa,income,age]])[0]
    gpa_n  = float(cont[0])

    n = len(student_embs)
    if   gpa_n>=0.66: seg = student_embs[int(n*0.66):]
    elif gpa_n>=0.33: seg = student_embs[int(n*0.33):int(n*0.66)]
    else:             seg = student_embs[:int(n*0.33)]
    stu_emb = seg.mean(axis=0)

    level     = profile.get("degree_level","bachelor")
    level_col = f"eligible_{level}"
    if level_col not in opto_df.columns: level_col = "eligible_bachelor"

    intl  = int(profile.get("International",0))
    cands = opto_df[opto_df[level_col]==1].copy()
    cands = cands[(cands["min_gpa_required"]<=0)|(cands["min_gpa_required"]/5.0<=gpa)]
    if intl:
        cands = cands[~cands["citizenship_required"].str.contains(
            "us_citizen|specific_residency",case=False,na=False)]
    if cands.empty: return cands.head(0)

    cands = cands.copy().reset_index(drop=True)

    # ── Field of study relevance score ─────────────────────────
    # Check if scholarship description/title mentions student's field
    user_field = str(profile.get("_field_of_study","") or
                     profile.get("field_of_study","")).lower().strip()

    if user_field and len(user_field) > 2:
        # Extract key field words (ignore short/common words)
        field_words = [w for w in user_field.split()
                       if len(w) > 3 and w not in
                       ["and","the","for","with","science","studies","management"]]

        def field_relevance(row):
            text = (str(row.get("scholarship_title","")) + " " +
                    str(row.get("description_cleaned","")) +
                    str(row.get("eligibility_summary",""))).lower()
            # Count how many field words appear
            matches = sum(1 for w in field_words if w in text)
            if matches > 0:
                return 1.0 + (matches * 0.3)   # boost relevant
            # Check for clearly irrelevant fields
            irrelevant = {
                "computer science": ["art","music","dance","nursing","law","social work","history","philosophy"],
                "engineering":      ["art","music","dance","nursing","law","social work","history"],
                "medicine":         ["art","music","engineering","law","business"],
                "law":              ["art","music","engineering","nursing","medicine"],
                "art":              ["engineering","nursing","medicine","law","computer"],
                "nursing":          ["art","music","engineering","law","computer"],
                "business":         ["art","music","nursing","medicine"],
                "social work":      ["engineering","computer","medicine","law"],
            }
            # Find matching irrelevant fields for user's field
            penalty_words = []
            for key, bad_list in irrelevant.items():
                if key in user_field:
                    penalty_words = bad_list
                    break
            if penalty_words:
                title_lower = str(row.get("scholarship_title","")).lower()
                if any(bad in title_lower for bad in penalty_words):
                    return 0.3   # heavy penalty for clearly wrong field
            return 0.8   # neutral — no field mentioned

        cands["field_relevance"] = cands.apply(field_relevance, axis=1)
    else:
        cands["field_relevance"] = 1.0
    type_enc = cands["scholarship_type"].fillna("Merit-Based").map(type_map).fillna(0).values.astype(np.float32)
    amount_n = cands["funding_amount_raw"].fillna(0).values.astype(np.float32)/(amt_max+1e-8)
    gpa_col  = np.full(len(cands),gpa_n,dtype=np.float32)
    inc_col  = np.full(len(cands),float(cont[1]),dtype=np.float32)
    deg_col  = np.full(len(cands),1 if level=="bachelor" else 0,dtype=np.float32)
    X = np.concatenate([np.tile(stu_emb,(len(cands),1)),
                        np.stack([type_enc,amount_n,gpa_col,inc_col,deg_col],axis=1)],axis=1)

    all_scores = []
    tf.eval()
    with torch.no_grad():
        for i in range(0,len(X),2048):
            b = torch.tensor(X[i:i+2048],dtype=torch.float32).to(DEVICE)
            all_scores.append(tf(b).cpu().numpy())
    scores = np.concatenate(all_scores).flatten()

    # Sigmoid → probabilities
    raw_tensor     = torch.tensor(scores, dtype=torch.float32)
    sigmoid_scores = torch.sigmoid(raw_tensor).numpy()

    # Multiply by field relevance BEFORE normalization
    # This is the key step — field mismatch gets heavily penalized
    field_rel = cands["field_relevance"].values
    combined  = sigmoid_scores * field_rel

    # Min-max normalize combined score
    # Rank-based normalization — always produces spread 40%–95%
    ranks = combined.argsort().argsort()
    norm  = 0.40 + (ranks / (len(ranks) - 1 + 1e-8)) * 0.55

    cands["transfer_score"] = norm
    cands["field_score"]    = field_rel   # store for transparency
    cands["_X_idx"]         = range(len(cands))
    st.session_state["_last_X"] = X
    return cands.sort_values("transfer_score",ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════

FEAT_NAMES = [
    "Collaborative","Collaborative","Collaborative","Collaborative",
    "Collaborative","Collaborative","Collaborative","Collaborative",
    "Collaborative","Collaborative","Collaborative","Collaborative",
    "Collaborative","Collaborative","Collaborative","Collaborative",
    "Scholarship Type","Funding Amount","Your GPA","Your Income","Degree Level",
]
INTERP_NAMES = ["Scholarship Type","Funding Amount","Your GPA","Your Income","Degree Level"]

@st.cache_data(show_spinner=False)
def compute_shap(_tf, _X_bytes, _device_str):
    try:
        import shap, torch, numpy as np
        X  = np.frombuffer(_X_bytes,dtype=np.float32).reshape(-1,21).copy()
        Xt = torch.tensor(X,dtype=torch.float32,requires_grad=True)
        bg = torch.tensor(X.mean(axis=0,keepdims=True),dtype=torch.float32)
        _tf.eval()
        exp = shap.GradientExplainer(_tf, bg)
        sv  = exp.shap_values(Xt)
        if isinstance(sv,list): sv=sv[0]
        if hasattr(sv,"detach"): sv=sv.detach().cpu().numpy()
        return np.array(sv)
    except Exception:
        try:
            import torch,numpy as np
            X   = np.frombuffer(_X_bytes,dtype=np.float32).reshape(-1,21).copy()
            X_t = torch.tensor(X,dtype=torch.float32,requires_grad=True)
            _tf(X_t).sum().backward()
            return (X_t.grad.detach().numpy()*X)
        except: return None


def render_shap(shap_row, profile: dict = None, row: dict = None):
    """
    Render interpretable SHAP explanation with human-readable text.
    Shows how each feature specifically impacted the recommendation.
    """
    import numpy as np

    # Use only interpretable features (indices 16-20)
    interp_vals  = shap_row[16:]  # [type, amount, gpa, income, degree]

    # Total absolute contribution for percentage calculation
    all_contribs = [abs(v) for v in interp_vals] + [1.0]  # +1 for field
    total        = sum(all_contribs) + 1e-8

    # Build human-readable explanations
    explanations = []

    # 1. Scholarship Type
    type_pct = abs(interp_vals[0]) / total * 100
    if profile and row:
        stype = str(row.get("scholarship_type","Merit-Based"))
        type_dir = "matches" if interp_vals[0] >= 0 else "doesn't match"
        explanations.append({
            "label":   "🏷️ Scholarship Type",
            "detail":  f"**{stype}** {type_dir} your profile pattern.",
            "pct":     type_pct,
            "positive": interp_vals[0] >= 0,
        })

    # 3. Funding Amount
    amt_pct = abs(interp_vals[1]) / total * 100
    if row:
        amount = float(row.get("funding_amount_raw", 0) or 0)
        amt_note = f"${amount:,.0f} award" if amount > 0 else "Amount unspecified"
        amt_dir  = "boosts" if interp_vals[1] >= 0 else "slightly reduces"
        explanations.append({
            "label":   "💰 Funding Amount",
            "detail":  f"{amt_note} — {amt_dir} your match score.",
            "pct":     amt_pct,
            "positive": interp_vals[1] >= 0,
        })

    # 4. GPA Match — most interpretable
    gpa_pct = abs(interp_vals[2]) / total * 100
    if profile and row:
        user_gpa = float(profile.get("final_gpa", 0) or 0)
        min_gpa  = float(row.get("min_gpa_required", 0) or 0) / 5.0
        if min_gpa > 0:
            if user_gpa >= min_gpa:
                gpa_detail = (f"Your GPA of **{user_gpa:.2f}** meets the "
                              f"{min_gpa:.1f} requirement "
                              f"(+{user_gpa-min_gpa:.2f} above threshold).")
                gpa_pos = True
            else:
                gpa_detail = (f"Your GPA of **{user_gpa:.2f}** is below the "
                              f"{min_gpa:.1f} requirement "
                              f"({min_gpa-user_gpa:.2f} gap).")
                gpa_pos = False
        else:
            gpa_detail = (f"No minimum GPA required — "
                          f"your **{user_gpa:.2f}** GPA is an advantage.")
            gpa_pos = interp_vals[2] >= 0
        explanations.append({
            "label":   "📊 GPA Match",
            "detail":  gpa_detail,
            "pct":     gpa_pct,
            "positive": gpa_pos,
        })

    # 5. Income / Financial Need
    inc_pct = abs(interp_vals[3]) / total * 100
    if profile and row:
        needs_fin = int(row.get("requires_financial_need", 0) or 0)
        user_need = int(profile.get("financial_need", 0) or 0)
        if needs_fin and user_need:
            inc_detail = "You demonstrate financial need — matches this scholarship's requirement."
            inc_pos    = True
        elif needs_fin and not user_need:
            inc_detail = "This scholarship requires financial need — you didn't indicate this."
            inc_pos    = False
        else:
            inc_detail = "No financial need requirement — open to all income levels."
            inc_pos    = interp_vals[3] >= 0
        explanations.append({
            "label":   "💵 Financial Need",
            "detail":  inc_detail,
            "pct":     inc_pct,
            "positive": inc_pos,
        })

    # 6. Degree Level
    deg_pct = abs(interp_vals[4]) / total * 100
    if profile:
        user_deg = str(profile.get("degree_level","bachelor")).title()
        deg_dir  = "matches" if interp_vals[4] >= 0 else "partially matches"
        explanations.append({
            "label":   "🎓 Degree Level",
            "detail":  f"Your **{user_deg}** level {deg_dir} this scholarship's eligibility.",
            "pct":     deg_pct,
            "positive": interp_vals[4] >= 0,
        })

    # 6. Field of Study
    if profile and row:
        user_field  = str(profile.get("_field_of_study","") or
                          profile.get("field_of_study","")).strip()
        sch_text    = (str(row.get("scholarship_title","")) + " " +
                       str(row.get("description_cleaned",""))).lower()
        field_score = float(row.get("field_score", 1.0) or 1.0)

        if user_field:
            field_words = [w for w in user_field.lower().split()
                           if len(w) > 3]
            matches = [w for w in field_words if w in sch_text]
            if matches:
                field_detail  = (f"Your field **{user_field}** is directly mentioned "
                                 f"in this scholarship's description — strong alignment.")
                field_positive = True
            elif field_score >= 1.0:
                field_detail  = (f"This scholarship is open to all fields including "
                                 f"**{user_field}** — no field restriction.")
                field_positive = True
            elif field_score < 0.5:
                field_detail  = (f"This scholarship appears targeted at a different field "
                                 f"than **{user_field}** — check eligibility carefully.")
                field_positive = False
            else:
                field_detail  = (f"No specific field requirement — "
                                 f"**{user_field}** students may apply.")
                field_positive = True

            field_pct = (abs(field_score - 0.8) / (total + 1e-8)) * 100 + 5
            explanations.append({
                "label":    "📚 Field of Study",
                "detail":   field_detail,
                "pct":      field_pct,
                "positive": field_positive,
            })

    # Sort by impact
    explanations.sort(key=lambda x: x["pct"], reverse=True)

    with st.expander("🔍 Why this match? (SHAP Explanation)"):
        st.caption("How each factor contributed to your match score:")
        st.markdown("")

        for exp in explanations:
            color = "#10b981" if exp["positive"] else "#ef4444"
            arrow = "▲" if exp["positive"] else "▼"
            pct   = exp["pct"]

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{exp['label']}**")
                st.markdown(
                    f"<span style='font-size:0.85rem;color:#4b5563'>"
                    f"{exp['detail']}</span>",
                    unsafe_allow_html=True
                )
                st.progress(float(min(pct/100, 1.0)))
            with col2:
                st.markdown(
                    f"<div style='text-align:center;padding-top:8px'>"
                    f"<div style='font-size:1.3rem;color:{color};font-weight:700'>{arrow}</div>"
                    f"<div style='font-size:0.9rem;font-weight:600;color:{color}'>"
                    f"{pct:.0f}%</div>"
                    f"<div style='font-size:0.7rem;color:#9ca3af'>of impact</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            st.markdown("")


# ══════════════════════════════════════════════════════════════
# GAP FINDER
# ══════════════════════════════════════════════════════════════

def run_gap_finder(profile, opto_df, top_n=10):
    import pandas as pd
    gpa  = float(profile.get("final_gpa") or 0)
    need = int(profile.get("financial_need",0))
    intl = int(profile.get("International",0))
    level= profile.get("degree_level","bachelor")
    lc   = f"eligible_{level}"
    if lc not in opto_df.columns: lc = "eligible_bachelor"
    gaps = []
    for _,sch in opto_df[opto_df[lc]==1].iterrows():
        items,score = [],0.0
        mgpa = float(sch.get("min_gpa_required",0) or 0)
        if mgpa>0:
            req = mgpa/5.0
            if req>gpa:
                diff = req-gpa
                if diff>0.8: continue
                items.append(f"Raise GPA by {diff:.2f} (need {req:.1f}, have {gpa:.2f})")
                score += diff
        if int(sch.get("requires_financial_need",0) or 0) and not need:
            items.append("Provide financial need documentation")
            score += 0.4
        cit = str(sch.get("citizenship_required","") or "").lower()
        if intl and ("us_citizen" in cit or "specific_residency" in cit): continue
        if items and score<=1.2:
            gaps.append({
                "scholarship_id":    sch.get("scholarship_id",""),
                "scholarship_title": sch["scholarship_title"],
                "scholarship_type":  sch.get("scholarship_type",""),
                "funding_amount":    float(sch.get("funding_amount_raw",0) or 0),
                "deadline":          str(sch.get("deadline_date","") or ""),
                "link":              str(sch.get("link","") or ""),
                "gap_score":         round(score,3),
                "action_items":      " | ".join(items),
            })
    if not gaps: return None
    return pd.DataFrame(gaps).sort_values(
        ["gap_score","funding_amount"],ascending=[True,False]).head(top_n)


# ══════════════════════════════════════════════════════════════
# GROQ CHATBOT
# ══════════════════════════════════════════════════════════════

def chat_with_groq(messages: list, profile: dict = None,
                    recs=None) -> str:
    """Call Groq API with Llama 3 — free tier."""
    if not GROQ_API_KEY:
        return _rule_based_fallback(messages, profile, recs)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        system = (
            "You are OptiScholar, an AI scholarship advisor. "
            "You help students find and apply for scholarships. "
            "Be concise, helpful, and encouraging. "
            "Answer in 2-3 sentences unless the user asks for details."
        )
        if profile and profile.get("final_gpa"):
            system += (
                f"The student has GPA {profile.get('final_gpa','?')}/4.0, "
                f"studying {profile.get('degree_level','?')} in "
                f"{profile.get('_field_of_study','unknown')}. "
            )
        if recs is not None and len(recs) > 0:
            top3 = recs.head(3)["scholarship_title"].tolist()
            system += f"Their top 3 matches are: {', '.join(top3)}. "

        # Only send last 6 messages to avoid token limits
        recent_messages = messages[-6:] if len(messages) > 6 else messages

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role":"system","content":system}] + recent_messages,
            max_tokens=400,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        # Show error in chat so we can debug
        error_msg = str(e)
        return f"⚠️ Groq API error: {error_msg}\n\nFalling back to rule-based response:\n\n" +                _rule_based_fallback(messages, profile, recs)


def _rule_based_fallback(messages, profile, recs):
    """Rule-based chatbot fallback."""
    try:
        from chatbot import OptiScholarChatbot
        if st.session_state.chatbot is None:
            st.session_state.chatbot = OptiScholarChatbot()
        bot = st.session_state.chatbot
        bot.set_context(profile=profile, recommendations=recs,
                         opto_df=st.session_state.opto_df)
        last = messages[-1]["content"] if messages else ""
        return bot.respond(last)
    except Exception:
        return "I'm here to help with scholarships! Try asking me to find scholarships or explain your eligibility."


# ══════════════════════════════════════════════════════════════
# PAGE: LOGIN / SIGNUP
# ══════════════════════════════════════════════════════════════

def page_auth():
    st.markdown("# 🎓 OptiScholar")
    st.markdown("#### *Your AI-Powered Scholarship Advisor*")
    st.divider()

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        tab1, tab2 = st.tabs(["🔑 Login","📝 Sign Up"])

        with tab1:
            with st.form("login_form"):
                st.markdown("### Welcome back!")
                email    = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login →", use_container_width=True):
                    ok, msg, user = DB.login(email, password)
                    if ok:
                        st.session_state.user    = user
                        st.session_state.profile = DB.get_profile(user["user_id"])
                        st.session_state.page    = "🏠 Dashboard"
                        st.success(f"Welcome back, {user['full_name'].split()[0]}!")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab2:
            with st.form("signup_form"):
                st.markdown("### Create your account")
                name     = st.text_input("Full Name", placeholder="Nada Mohamed")
                email    = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password (min 6 chars)", type="password")
                if st.form_submit_button("Create Account →", use_container_width=True):
                    ok, msg, uid = DB.create_user(email, password, name)
                    if ok:
                        _, _, user           = DB.login(email, password)
                        st.session_state.user    = user
                        st.session_state.profile = None
                        st.session_state.page    = "👤 My Profile"
                        st.success(f"Account created! Let's set up your profile.")
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("---")
        st.caption("🔒 Passwords are securely hashed. We never store plaintext.")


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:8px 0 16px'>
          <div style='font-family:"DM Serif Display",serif;
                      font-size:1.4rem'>🎓 OptiScholar</div>
          <div style='font-size:0.7rem;opacity:0.7;
                      text-transform:uppercase;letter-spacing:1px'>
            AI Scholarship Advisor</div>
        </div>""", unsafe_allow_html=True)

        if user:
            st.markdown(f"👋 **{user['full_name'].split()[0]}**")
            st.caption(user["email"])
            st.divider()

        pages = ["🏠 Dashboard","👤 My Profile","🎯 Recommendations",
                 "📋 My Scholarships","🗺️ Eligibility Roadmap","💬 Chatbot"]
        page  = st.radio("Navigation", pages,
                          index=pages.index(st.session_state.page)
                                if st.session_state.page in pages else 0,
                          label_visibility="collapsed")
        st.session_state.page = page
        st.divider()

        # Deadline notifications
        if user:
            reminders = DB.get_upcoming_reminders(user["user_id"], days_ahead=14)
            if reminders:
                st.markdown(f"🔔 **{len(reminders)} deadline{'s' if len(reminders)>1 else ''} soon!**")
                for r in reminders[:3]:
                    st.warning(f"⏰ {r['scholarship_title'][:30]}...\n{r['deadline_date']}")

        if user:
            if st.button("🚪 Log Out", use_container_width=True):
                for k in ["user","profile","recommendations","gap_results",
                           "_last_X","_shap_vals","chat_history"]:
                    st.session_state[k] = None \
                        if k not in ["chat_history"] else []
                st.rerun()

        st.divider()
        st.markdown("**System**")
        for name, path in [("DB",SCHOLARSHIPS_PATH),("Transfer Fn",TRANSFER_FN_PATH),("NCF",NCF_MODEL_PATH)]:
            st.markdown(f"{'🟢' if os.path.exists(path) else '🔴'} {name}")
        if st.session_state._models is not None:
            tf,_,_ = st.session_state._models
            st.markdown(f"{'🟢' if tf else '🔴'} Models loaded")


# ══════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════

def page_dashboard():
    user    = st.session_state.user
    profile = st.session_state.profile

    st.markdown(f"# 🏠 Welcome, {user['full_name'].split()[0]}!")
    st.divider()

    # Stats
    stats = DB.get_dashboard_stats(user["user_id"])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Matched Scholarships", stats["total_recommendations"] or "—")
    c2.metric("Applied",   stats["applied"])
    c3.metric("Saved",     stats["saved"])
    c4.metric("Reminders", stats["reminders"])

    st.markdown("<br>", unsafe_allow_html=True)

    # Profile completeness
    if not profile or not profile.get("final_gpa"):
        st.warning("⚠️ Your profile is incomplete. Complete it to get personalised recommendations.")
        if st.button("Complete My Profile →"):
            st.session_state.page = "👤 My Profile"; st.rerun()
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Your Profile")
        st.markdown(f"**GPA:** {profile.get('final_gpa','—')}/4.0")
        st.markdown(f"**Degree:** {str(profile.get('degree_level','')).title()}")
        st.markdown(f"**Field:** {profile.get('field_of_study','—')}")
        st.markdown(f"**Institution:** {profile.get('institution','—')}")
        if st.button("✏️ Edit Profile"):
            st.session_state.page = "👤 My Profile"; st.rerun()

    with col2:
        st.markdown("### Quick Actions")
        if st.button("🔍 Find Scholarships", use_container_width=True):
            st.session_state.page = "🎯 Recommendations"
            st.session_state.recommendations = None; st.rerun()
        if st.button("🗺️ View Eligibility Roadmap", use_container_width=True):
            st.session_state.page = "🗺️ Eligibility Roadmap"
            st.session_state.gap_results = None; st.rerun()
        if st.button("📋 My Saved Scholarships", use_container_width=True):
            st.session_state.page = "📋 My Scholarships"; st.rerun()

    # Recent interactions
    recent = DB.get_interactions(user["user_id"])
    if recent:
        st.divider()
        st.markdown("### Recent Activity")
        for r in recent[:5]:
            icon = {"applied":"✅","saved":"🔖","ignored":"❌"}.get(r["action"],"•")
            st.markdown(
                f"{icon} **{r['scholarship_title'][:55]}** — "
                f"*{r['action'].title()}* on {r['created_at'][:10]}"
            )


# ══════════════════════════════════════════════════════════════
# PAGE: MY PROFILE
# ══════════════════════════════════════════════════════════════

def page_profile():
    st.markdown("# 👤 My Profile")
    st.divider()

    user    = st.session_state.user
    profile = st.session_state.profile or {}

    tab1, tab2 = st.tabs(["📄 Upload Document","✏️ Manual Entry"])

    with tab1:
        st.markdown("### Upload Transcript or CV")
        st.info("PDF only. GPA, degree, age, and field extracted automatically using RoBERTa NER.")
        uploaded = st.file_uploader("Choose PDF", type=["pdf"], key="doc_upload")
        prefill  = {}
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name
            with st.spinner("🔍 Extracting profile with RoBERTa NER + Regex..."):
                result  = parse_document(tmp_path)
                prefill = prefill_from_parse(result)
                try: os.unlink(tmp_path)
                except: pass
            fields = result.get("fields",{})
            if fields:
                st.success(f"✅ Extracted {len(fields)} fields.")
                cols = st.columns(min(len(fields),4))
                for col,(fn,fd) in zip(cols,list(fields.items())[:4]):
                    icon = "🤖" if fd.get("source")=="roberta" else "📐"
                    col.metric(f"{icon} {fn.replace('_',' ').title()}",
                                str(fd["value"]),f"{fd['confidence']:.0%}")
        _profile_form(prefill or profile, key="upload")

    with tab2:
        _profile_form(profile, key="manual")


def _profile_form(prefill, key="form"):
    user = st.session_state.user
    with st.form(f"pf_{key}"):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Academic**")
            gpa    = st.number_input("GPA (0–4.0)",0.0,4.0,step=0.01,
                                      value=float(prefill.get("final_gpa") or 2.5),format="%.2f")
            degree = st.selectbox("Degree Level",
                                   ["bachelor","master","phd","high_school"],
                                   index=int(prefill.get("degree_level_idx",0)))
            field  = st.text_input("Field of Study",
                                    value=str(prefill.get("field_of_study") or prefill.get("_field_of_study","") or ""))
            inst   = st.text_input("Institution",
                                    value=str(prefill.get("institution","") or ""))
        with c2:
            st.markdown("**Personal**")
            age    = st.number_input("Age",13,35,step=1,
                                      value=int(prefill.get("age") or 20))
            gender = st.selectbox("Gender",["Male","Female"],
                                   index=int(prefill.get("gender_idx",0)))
            ses    = st.selectbox("Socioeconomic Status",
                                   ["Low","Middle","High"],index=1)
            income = st.number_input("Household Income (USD/yr)",
                                      0,3000000,step=1000,value=int(prefill.get("household_income",50000) or 50000))
        c3,c4 = st.columns(2)
        with c3: need = st.checkbox("Financial need",value=bool(prefill.get("financial_need",0)))
        with c4: intl = st.checkbox("International student",value=bool(prefill.get("International",0) or prefill.get("is_international",0)))

        if st.form_submit_button("💾 Save Profile",use_container_width=True):
            new_profile = {
                "final_gpa":        gpa, "gpa_proxy": gpa,
                "degree_level":     degree, "age": age,
                "gender":           gender, "ses_category": ses,
                "household_income": income,
                "financial_need":   int(need),
                "International":    int(intl),
                "_field_of_study":  field,
                "_institution":     inst,
                "field_of_study":   field,
                "institution":      inst,
            }
            st.session_state.profile         = new_profile
            st.session_state.recommendations = None
            st.session_state.gap_results     = None
            DB.save_profile(user["user_id"], new_profile)
            st.success("✅ Profile saved!")
            st.balloons()


# ══════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════

def page_recommendations():
    st.markdown("# 🎯 Scholarship Recommendations")
    st.divider()

    user    = st.session_state.user
    profile = st.session_state.profile

    if not profile or not profile.get("final_gpa"):
        st.warning("Please complete your profile first.")
        if st.button("Go to Profile →"):
            st.session_state.page="👤 My Profile"; st.rerun()
        return

    if st.session_state._models is None:
        st.error("Models not loaded yet.")
        return

    opto_df = st.session_state.opto_df

    with st.expander("📋 Your Profile",expanded=False):
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("GPA",f"{profile.get('final_gpa',0):.2f}")
        c2.metric("Level",str(profile.get("degree_level","")).title())
        c3.metric("Fin Need","Yes" if profile.get("financial_need") else "No")
        c4.metric("Intl","Yes" if profile.get("International") else "No")

    c1,c2 = st.columns([1,3])
    with c1: top_n = st.selectbox("Show top",[10,20,30,50],index=1)
    with c2: run   = st.button("🔍 Find My Scholarships",use_container_width=True)

    if run:
        st.session_state.recommendations = None
        st.session_state["_shap_vals"]   = None

    if run or st.session_state.recommendations is not None:
        if st.session_state.recommendations is None:
            pb = st.progress(0,"Filtering eligible scholarships...")
            try:
                pb.progress(30,"Running Transfer Function (NCF → TF)...")
                recs = run_recommendations(profile, opto_df, top_n)

                pb.progress(65,"Computing SHAP explanations...")
                shap_vals = None
                X = st.session_state.get("_last_X")
                if X is not None:
                    tf,_,enc = st.session_state["_models"]
                    try:
                        top_idx  = recs["_X_idx"].values
                        X_top    = X[top_idx].astype("float32")
                        shap_vals= compute_shap(tf,X_top.tobytes(),str(enc["DEVICE"]))
                    except: pass

                st.session_state.recommendations = recs
                st.session_state["_shap_vals"]   = shap_vals

                # Cache to DB + add deadline reminders
                recs_list = recs.to_dict("records")
                DB.save_recommendations(user["user_id"], recs_list)
                for r in recs_list[:10]:
                    dl = str(r.get("deadline_date","") or "")
                    if dl and dl not in ["","nan","None"]:
                        DB.add_reminder(user["user_id"], r)

                pb.progress(100,"✅ Done!")
                import time; time.sleep(0.4); pb.empty()

            except Exception as e:
                pb.empty()
                st.error(f"Error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

        recs      = st.session_state.recommendations
        shap_vals = st.session_state.get("_shap_vals")

        if recs is None or len(recs)==0:
            st.warning("No scholarships matched. Try adjusting your profile.")
            return

        st.markdown(f"### {len(recs):,} scholarships matched your profile")
        st.caption("Ranked by Transfer Function (AUC 82.4%) · 🔍 Click 'Why this match?' for SHAP explanation")
        st.divider()

        import numpy as np

        for i,(_, row) in enumerate(recs.iterrows()):
            score    = max(0.0,min(1.0,float(row.get("transfer_score",0) or 0)))
            title    = str(row.get("scholarship_title",""))
            stype    = str(row.get("scholarship_type",""))
            amount   = float(row.get("funding_amount_raw",0) or 0)
            desc     = str(row.get("description_cleaned",""))[:300]
            deadline = str(row.get("deadline_date","") or "").strip()
            link     = str(row.get("link","") or "").strip()
            sch_id   = str(row.get("scholarship_id","") or f"SCH_{i}")

            if deadline in ["","nan","None","[NOT_AVAILABLE]"]: deadline=""
            if link     in ["","nan","None","[NO_LINK]"]:       link=""

            badge = f"⭐ #{i+1} Best Match" if i<3 else f"#{i+1}"

            # Get current user action for this scholarship
            current_action = DB.get_interaction_action(
                user["user_id"], sch_id
            )
            action_icons = {"applied":"✅ Applied","saved":"🔖 Saved",
                             "ignored":"❌ Ignored", None:"Not reviewed"}

            with st.container(border=True):
                col_main, col_score = st.columns([4,1])
                with col_main:
                    st.markdown(f"### {badge} — {title}")
                    if desc:
                        st.caption(f"{desc}{'...' if len(desc)==300 else ''}")
                    meta = []
                    if stype:    meta.append(f"🏷️ **{stype}**")
                    if amount:   meta.append(f"💰 **${amount:,.0f}**")
                    if deadline: meta.append(f"⏰ {deadline}")
                    if meta: st.markdown("  ·  ".join(meta))
                    if link: st.link_button("🔗 Apply Now", link)

                with col_score:
                    st.metric("Match",f"{score:.1%}")
                    if current_action:
                        st.markdown(
                            f"<div style='font-size:0.75rem;text-align:center;"
                            f"color:{'#10b981' if current_action=='applied' else '#f59e0b' if current_action=='saved' else '#6b7280'}'>"
                            f"{action_icons[current_action]}</div>",
                            unsafe_allow_html=True
                        )

                # Action buttons
                ac1,ac2,ac3 = st.columns(3)
                with ac1:
                    if st.button("✅ Applied",key=f"app_{i}_{sch_id[:8]}",
                                  use_container_width=True):
                        DB.save_interaction(user["user_id"],sch_id,"applied",dict(row))
                        st.toast(f"Marked as Applied! 🎉"); st.rerun()
                with ac2:
                    if st.button("🔖 Save",key=f"sav_{i}_{sch_id[:8]}",
                                  use_container_width=True):
                        DB.save_interaction(user["user_id"],sch_id,"saved",dict(row))
                        st.toast(f"Saved! 🔖"); st.rerun()
                with ac3:
                    if st.button("❌ Ignore",key=f"ign_{i}_{sch_id[:8]}",
                                  use_container_width=True):
                        DB.save_interaction(user["user_id"],sch_id,"ignored",dict(row))
                        st.toast("Ignored."); st.rerun()

                # SHAP
                if shap_vals is not None and i<len(shap_vals):
                    render_shap(shap_vals[i],
                                profile=st.session_state.profile,
                                row=dict(row))

            st.markdown("")


# ══════════════════════════════════════════════════════════════
# PAGE: MY SCHOLARSHIPS
# ══════════════════════════════════════════════════════════════

def page_my_scholarships():
    user = st.session_state.user
    st.markdown("# 📋 My Scholarships")
    st.divider()

    tab1,tab2,tab3 = st.tabs(["✅ Applied","🔖 Saved","❌ Ignored"])

    for tab,action,icon in [(tab1,"applied","✅"),(tab2,"saved","🔖"),(tab3,"ignored","❌")]:
        with tab:
            items = DB.get_interactions(user["user_id"], action=action)
            if not items:
                st.info(f"No {action} scholarships yet.")
                continue
            for item in items:
                with st.container(border=True):
                    c1,c2 = st.columns([4,1])
                    with c1:
                        st.markdown(f"### {icon} {item['scholarship_title']}")
                        meta = []
                        if item.get("scholarship_type"): meta.append(f"🏷️ {item['scholarship_type']}")
                        if item.get("amount"):           meta.append(f"💰 ${item['amount']:,.0f}")
                        if item.get("deadline"):         meta.append(f"⏰ {item['deadline']}")
                        if meta: st.markdown("  ·  ".join(meta))
                        if item.get("link"): st.link_button("🔗 Apply",item["link"])
                        st.caption(f"Updated: {item['created_at'][:10]}")
                    with c2:
                        if item.get("match_score"):
                            st.metric("Match",f"{item['match_score']:.0%}")


# ══════════════════════════════════════════════════════════════
# PAGE: ELIGIBILITY ROADMAP
# ══════════════════════════════════════════════════════════════

def generate_general_advice(profile: dict) -> list:
    """Generate general scholarship improvement advice for any student."""
    gpa    = float(profile.get("final_gpa") or 0)
    field  = str(profile.get("_field_of_study","") or profile.get("field_of_study","") or "").lower()
    degree = str(profile.get("degree_level","bachelor")).lower()
    need   = int(profile.get("financial_need",0) or 0)

    advice = [
        {
            "icon": "🤝",
            "title": "Volunteer & Community Service",
            "text": "Engage in community service such as volunteering at local organizations, "
                    "participating in charity events, or joining a community club. "
                    "Community Service scholarships have the highest approval rates (58.6%) in our dataset.",
        },
        {
            "icon": "👑",
            "title": "Leadership Roles",
            "text": "Take on leadership positions in extracurricular activities — team captain, "
                    "club president, or event organizer. Many Merit-Based scholarships "
                    "explicitly reward demonstrated leadership.",
        },
        {
            "icon": "🔬",
            "title": "Research Experience",
            "text": "Participate in research projects, internships, or academic competitions "
                    "to demonstrate your skills. Research experience significantly "
                    "boosts eligibility for Academic Excellence scholarships.",
        },
        {
            "icon": "🌍",
            "title": "Language Skills",
            "text": "Develop proficiency in multiple languages to enhance your profile. "
                    "Bilingual students have access to additional international scholarship pools.",
        },
        {
            "icon": "🏆",
            "title": "Awards & Recognition",
            "text": "Strive to receive awards, recognition, or academic publications. "
                    "Documented achievements strengthen any scholarship application.",
        },
        {
            "icon": "🤝",
            "title": "Networking",
            "text": "Establish relationships with professionals in your field through "
                    "networking events, mentorship programs, or industry conferences. "
                    "Many company-sponsored scholarships prefer connected candidates.",
        },
        {
            "icon": "💡",
            "title": "Soft Skills",
            "text": "Develop communication, teamwork, time management, and problem-solving "
                    "skills through workshops or extracurricular activities. "
                    "These are explicitly evaluated in personal statement scholarships.",
        },
        {
            "icon": "💼",
            "title": "Projects & Portfolio",
            "text": "Build a portfolio of personal projects to showcase your skills. "
                    "Tangible work samples set you apart in competitive scholarship pools.",
        },
    ]

    # Field-specific advice
    if "computer" in field or "software" in field or "data" in field or "ai" in field:
        advice.insert(2, {
            "icon": "💻",
            "title": "Technical Projects (CS-specific)",
            "text": "Build open-source projects on GitHub, participate in hackathons, "
                    "or contribute to research in AI/ML. Many CS scholarships specifically "
                    "require a portfolio of technical work.",
        })
    elif "engineering" in field:
        advice.insert(2, {
            "icon": "⚙️",
            "title": "Engineering Projects (field-specific)",
            "text": "Participate in engineering competitions (e.g. IEEE, robotics clubs) "
                    "and build hands-on project experience. STEM scholarships highly value "
                    "demonstrated engineering problem-solving.",
        })
    elif "medicine" in field or "nursing" in field or "health" in field:
        advice.insert(2, {
            "icon": "🏥",
            "title": "Clinical Experience (field-specific)",
            "text": "Gain clinical volunteering or shadowing experience. Healthcare "
                    "scholarships almost universally require demonstrated patient-facing experience.",
        })
    elif "business" in field or "management" in field or "finance" in field:
        advice.insert(2, {
            "icon": "📊",
            "title": "Business Experience (field-specific)",
            "text": "Complete internships, case competitions, or start a small venture. "
                    "Business scholarships reward entrepreneurial and analytical thinking.",
        })

    # GPA-specific advice
    if gpa < 3.0:
        advice.insert(0, {
            "icon": "📈",
            "title": "GPA Improvement (Priority)",
            "text": f"Your GPA of {gpa:.2f} currently limits Merit-Based scholarship access. "
                    "Focus on your next semester — improving to 3.0+ unlocks hundreds of "
                    "additional scholarships. Consider tutoring, study groups, or office hours.",
        })
    elif gpa < 3.5:
        advice.insert(0, {
            "icon": "📈",
            "title": "GPA Improvement",
            "text": f"With a {gpa:.2f} GPA, improving to 3.5+ would unlock Academic Excellence "
                    "scholarships with lower competition. Focus on your strongest subjects first.",
        })

    # Financial need advice
    if not need:
        advice.append({
            "icon": "💵",
            "title": "Financial Need Documentation",
            "text": "If your household income qualifies, consider documenting financial need. "
                    "Need-Based scholarships have the largest award amounts ($5,000+) "
                    "and significantly less competition than merit scholarships.",
        })

    return advice


def page_roadmap():
    st.markdown("# 🗺️ Eligibility Roadmap")
    st.markdown("Personalized guidance to improve your scholarship eligibility.")
    st.divider()

    profile = st.session_state.profile
    if not profile or not profile.get("final_gpa"):
        st.warning("Please complete your profile first.")
        if st.button("Go to Profile →"):
            st.session_state.page = "👤 My Profile"; st.rerun()
        return

    gpa    = float(profile.get("final_gpa") or 0)
    field  = str(profile.get("_field_of_study","") or profile.get("field_of_study","") or "your field")
    degree = str(profile.get("degree_level","bachelor")).title()

    # ── Section 1: General Advice ─────────────────────────────
    st.markdown("## 🎓 General Profile Improvement")
    st.info(
        f"To increase your eligibility for most scholarships, "
        f"consider improving the following areas. "
        f"These apply to all students regardless of field or GPA."
    )

    advice_list = generate_general_advice(profile)
    for i, adv in enumerate(advice_list, 1):
        with st.container(border=True):
            c1, c2 = st.columns([1, 10])
            with c1:
                st.markdown(f"<div style='font-size:2rem;text-align:center'>{adv['icon']}</div>",
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{i}. {adv['title']}**")
                st.markdown(adv["text"])

    # ── Section 2: Personalized summary ───────────────────────
    st.divider()
    st.markdown("## 👤 For Your Specific Profile")

    personalized = []
    if gpa < 3.5:
        personalized.append(
            f"📊 With a **{gpa:.2f} GPA**, focus on improving your grades — "
            f"reaching 3.5 unlocks Academic Excellence scholarships with less competition."
        )
    if gpa >= 3.5:
        personalized.append(
            f"⭐ Your **{gpa:.2f} GPA** is excellent — prioritize Academic Excellence "
            f"scholarships which have less competition despite higher requirements."
        )

    field_lower = field.lower()
    if "computer" in field_lower or "software" in field_lower:
        personalized.append(
            f"💻 As a **{field}** student, emphasize GitHub projects, hackathon wins, "
            f"and technical research. These are the top differentiators for CS scholarships."
        )
    elif "engineering" in field_lower:
        personalized.append(
            f"⚙️ As an **{field}** student, join IEEE or engineering clubs and "
            f"document any lab/project experience — STEM scholarships reward this highly."
        )
    elif "business" in field_lower or "management" in field_lower:
        personalized.append(
            f"📈 As a **{field}** student, internship experience and case competition "
            f"results are your strongest differentiators for business scholarships."
        )
    elif "nursing" in field_lower or "medicine" in field_lower or "health" in field_lower:
        personalized.append(
            f"🏥 As a **{field}** student, clinical volunteering and patient-care hours "
            f"are mandatory for most healthcare scholarships."
        )
    else:
        personalized.append(
            f"📚 As a **{field}** student, highlight relevant coursework, research, "
            f"and any published or presented academic work in your applications."
        )

    if not int(profile.get("financial_need",0)):
        personalized.append(
            "💵 Consider documenting **financial need** if applicable — "
            "Need-Based scholarships offer the largest awards with less competition."
        )

    personalized.append(
        "🤝 Highlight **leadership roles, volunteer work, and community service** "
        "in every application — these are evaluated across all scholarship types."
    )

    for tip in personalized:
        st.markdown(f"• {tip}")

    # ── Section 3: Near-miss scholarships ─────────────────────
    st.divider()
    st.markdown("## 🎯 Scholarships You're Close To Qualifying For")

    c1,c2 = st.columns([1,3])
    with c1: top_n = st.selectbox("Show top",[5,10,15,20],index=1)
    with c2: run   = st.button("🔍 Find Near-Miss Scholarships",use_container_width=True)

    if run: st.session_state.gap_results = None

    if run or st.session_state.gap_results is not None:
        if st.session_state.gap_results is None:
            with st.spinner("Scanning scholarships for near-misses..."):
                gaps = run_gap_finder(profile, st.session_state.opto_df, top_n)
                st.session_state.gap_results = gaps

        gaps = st.session_state.gap_results
        if gaps is None or len(gaps)==0:
            st.success("🎉 You qualify for all nearby scholarships! Check Recommendations.")
            return

        st.markdown(f"**{len(gaps)} scholarships** you could qualify for with small improvements:")
        st.markdown("")

        user = st.session_state.user
        for i,(_,row) in enumerate(gaps.iterrows(),1):
            title    = str(row.get("scholarship_title",""))
            stype    = str(row.get("scholarship_type",""))
            amount   = float(row.get("funding_amount",0) or 0)
            gap      = float(row.get("gap_score",0) or 0)
            actions  = str(row.get("action_items",""))
            deadline = str(row.get("deadline","") or "").strip()
            link     = str(row.get("link","") or "").strip()
            sch_id   = str(row.get("scholarship_id","") or f"GAP_{i}")
            if deadline in ["","nan","None"]: deadline=""
            if link    in ["","nan","None"]: link=""

            label    = ("🟢 Very Close" if gap<=0.3 else
                        "🟡 Close"      if gap<=0.7 else "🔴 Needs Work")

            with st.container(border=True):
                c1,c2 = st.columns([4,1])
                with c1:
                    st.markdown(f"### {i}. {title}")
                    meta = [f"🏷️ {stype}", f"💰 ${amount:,.0f}"]
                    if deadline: meta.append(f"⏰ {deadline}")
                    st.markdown("  ·  ".join(meta))
                    st.success(f"✅ **What to do:** {actions}")
                    if link: st.link_button("🔗 Apply", link)
                with c2:
                    st.metric("Gap Score", f"{gap:.2f}")
                    st.markdown(label)
                    if st.button("🔖 Save",key=f"gsav_{i}_{sch_id[:6]}",
                                  use_container_width=True):
                        DB.save_interaction(
                            user["user_id"], sch_id, "saved",
                            {"scholarship_title":title,"funding_amount_raw":amount,
                             "deadline_date":deadline,"link":link,"transfer_score":0}
                        )
                        st.toast("Saved! 🔖")


# ══════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ══════════════════════════════════════════════════════════════

def page_chatbot():
    st.markdown("# 💬 OptiScholar Assistant")
    st.caption("Powered by Llama 3 via Groq API" if GROQ_API_KEY else "Rule-based mode (add GROQ_API_KEY for AI responses)")
    st.divider()

    if GROQ_API_KEY:
        st.caption("🤖")
    else:
        st.info(
            "💡 **Enable AI chatbot:** Get a free API key at "
            "[console.groq.com](https://console.groq.com) and set it with:\n"
            "```bash\nexport GROQ_API_KEY=your_key_here\n```\n"
            "Restart the app after setting it."
        )

    # Chat history
    if not st.session_state.chat_history:
        st.session_state.chat_history = []
        welcome = ("Hi! How can I help you today? "
                   "Ask me about scholarships, application tips, or your eligibility gaps!")
        st.session_state.chat_history.append({"role":"assistant","content":welcome})

    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"]=="user" else "chat-bot"
        pfx = "" if msg["role"]=="user" else "🎓 "
        st.markdown(
            f"<div class='{cls}'>{pfx}{msg['content']}</div>"
            "<div style='clear:both'></div>",
            unsafe_allow_html=True
        )

    # st.chat_input: clears automatically, supports Enter key
    user_input = st.chat_input("Ask about scholarships...")

    if user_input and user_input.strip():
        st.session_state.chat_history.append(
            {"role":"user","content":user_input.strip()}
        )
        with st.spinner("Thinking..."):
            response = chat_with_groq(
                st.session_state.chat_history,
                profile=st.session_state.profile,
                recs=st.session_state.recommendations,
            )
        st.session_state.chat_history.append(
            {"role":"assistant","content":response}
        )
        st.rerun()

    if len(st.session_state.chat_history) > 1:
        if st.button("🗑️ Clear conversation",key="clr"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    init_session()
    inject_css()

    # Not logged in → show auth page
    if st.session_state.user is None:
        page_auth()
        return

    # Load data & models once
    if st.session_state.opto_df is None:
        if os.path.exists(SCHOLARSHIPS_PATH):
            with st.spinner("Loading scholarship database..."):
                st.session_state.opto_df = load_scholarships()
        else:
            st.error(f"Dataset not found: {SCHOLARSHIPS_PATH}"); st.stop()

    if st.session_state._models is None:
        if os.path.exists(TRANSFER_FN_PATH):
            with st.spinner("🔄 Loading AI models..."):
                st.session_state._models = load_models()

    if st.session_state._ner_pipeline is None:
        with st.spinner("🔄 Loading RoBERTa NER..."):
            try: st.session_state._ner_pipeline = load_ner()
            except Exception as e: st.warning(f"NER: {e}")

    render_sidebar()

    page = st.session_state.page
    if   page == "🏠 Dashboard":           page_dashboard()
    elif page == "👤 My Profile":           page_profile()
    elif page == "🎯 Recommendations":      page_recommendations()
    elif page == "📋 My Scholarships":      page_my_scholarships()
    elif page == "🗺️ Eligibility Roadmap":  page_roadmap()
    elif page == "💬 Chatbot":              page_chatbot()


if __name__ == "__main__":
    main()
