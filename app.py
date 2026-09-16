
import os
import json
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from ingestion import load_transactions, load_call_logs, load_chat_logs
from correlation import build_entity_edges, compute_entity_features
from risk_model import load_or_train_model, load_training_data, score_entities
from graph_viz import build_graph, render_static_graph


# ============================================================
# FRAUD-LINK AI — Interactive Streamlit Frontend
# ============================================================

st.set_page_config(
    page_title="FRAUD-LINK AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------- CSS ------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(99,102,241,.16), transparent 28%),
        radial-gradient(circle at 10% 30%, rgba(6,182,212,.10), transparent 25%),
        #080b14;
    color: #eef2ff;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #080b14 100%);
    border-right: 1px solid rgba(148,163,184,.14);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb;
}

.hero {
    padding: 28px 30px;
    border: 1px solid rgba(99,102,241,.30);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(30,41,59,.82), rgba(15,23,42,.70)),
        radial-gradient(circle at 95% 10%, rgba(139,92,246,.22), transparent 30%);
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 38px;
    font-weight: 700;
    letter-spacing: -1.5px;
    margin-bottom: 5px;
}

.hero-sub {
    color: #a5b4fc;
    font-size: 15px;
}

.badge {
    display: inline-block;
    padding: 6px 11px;
    margin: 5px 5px 0 0;
    border-radius: 999px;
    background: rgba(99,102,241,.13);
    border: 1px solid rgba(129,140,248,.25);
    color: #c7d2fe;
    font-size: 12px;
}

.metric-card {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 18px;
    padding: 17px 18px;
    min-height: 112px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.metric-value {
    font-size: 29px;
    font-weight: 700;
    margin-top: 8px;
    color: #f8fafc;
}

.metric-note {
    color: #64748b;
    font-size: 11px;
    margin-top: 4px;
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    margin: 25px 0 10px;
}

.panel {
    background: rgba(15,23,42,.58);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 18px;
    padding: 18px;
}

.risk-high {
    color: #fb7185;
    font-weight: 700;
}

.risk-medium {
    color: #fbbf24;
    font-weight: 700;
}

.risk-low {
    color: #34d399;
    font-weight: 700;
}

.timeline {
    border-left: 2px solid #6366f1;
    margin-left: 10px;
    padding-left: 18px;
}

.timeline-item {
    margin-bottom: 16px;
}

.timeline-time {
    font-family: 'JetBrains Mono', monospace;
    color: #818cf8;
    font-size: 11px;
}

.timeline-text {
    color: #e2e8f0;
    margin-top: 3px;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(129,140,248,.28);
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 600;
    min-height: 44px;
}

.stButton > button:hover {
    border-color: #a5b4fc;
    box-shadow: 0 0 22px rgba(99,102,241,.28);
}

div[data-testid="stFileUploader"] {
    border-radius: 15px;
}

div[data-testid="stDataFrame"] {
    border-radius: 14px;
}

.small-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #94a3b8;
}

</style>
""", unsafe_allow_html=True)


# ---------------------- Session state -----------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ---------------------- Helper functions --------------------

def save_uploaded_file(uploaded_file):
    """Save a Streamlit UploadedFile temporarily."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level):
    cls = {
        "High": "risk-high",
        "Medium": "risk-medium",
        "Low": "risk-low",
    }.get(level, "")
    return f'<span class="{cls}">{level}</span>'


def make_risk_table(risk_df):
    if risk_df.empty:
        return risk_df

    display = risk_df.copy()

    display["risk_score"] = display["risk_score"].round(1)

    rename_map = {
        "entity": "Entity",
        "entity_type": "Type",
        "risk_score": "Risk Score",
        "risk_level": "Risk Level",
        "transaction_count": "Transactions",
        "total_amount": "Amount Involved",
        "linked_accounts": "Linked Accounts",
        "linked_phone_numbers": "Linked Phones",
        "sim_changes": "Device/SIM Changes",
        "transaction_velocity": "Velocity",
    }

    display = display.rename(columns=rename_map)

    preferred = [
        "Entity", "Type", "Risk Score", "Risk Level",
        "Transactions", "Amount Involved",
        "Linked Accounts", "Linked Phones",
        "Device/SIM Changes", "Velocity"
    ]

    return display[[c for c in preferred if c in display.columns]]


def create_json_report(analysis):
    """Create a lightweight JSON investigative brief."""
    risk_df = analysis["risk_df"].copy()

    records = risk_df.to_dict(orient="records")

    report = {
        "case": "FRAUD-LINK AI",
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "files_processed": analysis["files_processed"],
            "entities_found": len(risk_df),
            "relationships_found": len(analysis["edges"]),
            "high_risk_entities": int(
                (risk_df["risk_level"] == "High").sum()
            ) if not risk_df.empty else 0,
        },
        "risk_entities": records,
    }

    return json.dumps(report, indent=2, default=str)


# --------------------------- Sidebar -------------------------

with st.sidebar:
    st.markdown("## 🛡️ FRAUD-LINK AI")
    st.caption("Cyber Fraud Intelligence Console")

    st.markdown("---")

    st.markdown("### 📁 Evidence Sources")

    transaction_file = st.file_uploader(
        "Bank / UPI Transactions",
        type=["csv", "xlsx", "xls", "json", "txt"],
        key="transaction_file",
    )

    call_file = st.file_uploader(
        "CDR / IPDR Call Records",
        type=["csv", "xlsx", "xls", "json", "txt"],
        key="call_file",
    )

    chat_file = st.file_uploader(
        "Chat / Communication Logs",
        type=["csv", "xlsx", "xls", "json", "txt"],
        key="chat_file",
    )

    training_file = st.file_uploader(
        "Optional labeled training data",
        type=["csv", "xlsx", "xls"],
        help="Columns required: six model features plus is_fraud (0 or 1).",
        key="training_file",
    )

    st.markdown("---")

    st.markdown("### ⚙️ Analysis Mode")

    analysis_mode = st.selectbox(
        "Select investigation mode",
        [
            "Full Correlation",
            "Transaction Focus",
            "Communication Focus",
        ],
    )

    show_raw = st.checkbox("Show processed data", value=False)
    show_graph = st.checkbox("Show network graph", value=True)

    st.markdown("---")
    st.caption("⚡ Lightweight • Explainable • Investigator-focused")
    st.caption("SHA-256 evidence integrity enabled")


# ---------------------------- Hero ---------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ FRAUD-LINK AI</div>
        <div class="hero-sub">
            AI-Powered Unified Cyber Fraud Analysis & Digital Evidence Correlator
        </div>
        <div style="margin-top:14px;">
            <span class="badge">Pandas</span>
            <span class="badge">Decision Tree</span>
            <span class="badge">NetworkX</span>
            <span class="badge">SHA-256</span>
            <span class="badge">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------- Empty state --------------------------

files_uploaded = any([
    transaction_file,
    call_file,
    chat_file,
])

if not files_uploaded:
    st.markdown(
        """
        <div class="panel">
            <h3>🚨 Golden-Hour Investigation Console</h3>
            <p style="color:#94a3b8;">
            Upload fragmented cyber-fraud evidence from the sidebar.
            FRAUD-LINK AI will normalize the records, correlate entities,
            calculate explainable risk, and construct a fraud network.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### How the pipeline works")

    c1, c2, c3, c4, c5 = st.columns(5)

    steps = [
        ("01", "INGEST", "Read evidence"),
        ("02", "NORMALIZE", "Clean entities"),
        ("03", "CORRELATE", "Find links"),
        ("04", "SCORE", "Detect risk"),
        ("05", "VISUALIZE", "Build network"),
    ]

    for col, (num, title, desc) in zip([c1, c2, c3, c4, c5], steps):
        with col:
            st.markdown(
                f"""
                <div class="panel" style="text-align:center;">
                    <div style="font-size:12px;color:#818cf8;">{num}</div>
                    <div style="font-weight:700;margin-top:7px;">{title}</div>
                    <div style="font-size:11px;color:#94a3b8;margin-top:5px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info("Upload at least one evidence file to start an investigation.")
    st.stop()


# --------------------- Analyze button ------------------------

st.markdown('<div class="section-title">🔎 Investigation Control</div>',
            unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([2, 1, 1])

with col_a:
    st.write(
        f"Evidence loaded: "
        f"**{sum(bool(x) for x in [transaction_file, call_file, chat_file])} source(s)**"
    )

with col_b:
    analyze_clicked = st.button(
        "🚀 ANALYZE CASE",
        use_container_width=True,
    )

with col_c:
    clear_clicked = st.button(
        "↻ RESET",
        use_container_width=True,
    )

if clear_clicked:
    st.session_state.analysis_done = False
    st.session_state.analysis = None
    st.rerun()


# ------------------------- Pipeline ---------------------------

if analyze_clicked:
    with st.status("Running FRAUD-LINK AI correlation pipeline...", expanded=True) as status:
        evidence = {}
        processed_preview = {}

        # 1. Ingestion
        st.write("📥 Ingesting evidence files...")

        if transaction_file:
            path = save_uploaded_file(transaction_file)
            df = load_transactions(path)
            evidence["transactions"] = df
            processed_preview["transactions"] = df

        if call_file:
            path = save_uploaded_file(call_file)
            df = load_call_logs(path)
            evidence["call_logs"] = df
            processed_preview["call_logs"] = df

        if chat_file:
            path = save_uploaded_file(chat_file)
            df = load_chat_logs(path)
            evidence["chat_logs"] = df
            processed_preview["chat_logs"] = df

        # 2. Correlation
        st.write("🔗 Correlating phones, accounts, UPI IDs, devices and IPs...")

        edges = build_entity_edges(evidence)
        features = compute_entity_features(edges)

        # 3. Risk model
        st.write("🧠 Applying Decision Tree risk scoring...")

        training_df = None
        training_source = None
        if training_file:
            training_path = save_uploaded_file(training_file)
            training_df = load_training_data(training_path)
            training_source = training_file.name

        model, report = load_or_train_model(training_df, training_source)
        risk_df = score_entities(features, model)

        # 4. Graph
        st.write("🕸️ Constructing fraud network...")

        entity_risk = risk_df[["entity", "risk_level", "risk_score"]].copy() if not risk_df.empty else pd.DataFrame(columns=["entity", "risk_level", "risk_score"])
        graph = build_graph(edges, entity_risk)

        os.makedirs("output", exist_ok=True)

        graph_path = "output/network_graph.png"
        render_static_graph(graph, graph_path)

        st.session_state.analysis = {
            "evidence": evidence,
            "processed_preview": processed_preview,
            "edges": edges,
            "features": features,
            "risk_df": risk_df,
            "graph": graph,
            "model_report": report,
            "files_processed": len(evidence),
            "graph_path": graph_path,
        }

        st.session_state.analysis_done = True

        status.update(
            label="✅ Investigation analysis completed",
            state="complete",
            expanded=False,
        )


# ----------------------- Dashboard ---------------------------

if not st.session_state.analysis_done:
    st.stop()

analysis = st.session_state.analysis
risk_df = analysis["risk_df"]
edges = analysis["edges"]

st.markdown('<div class="section-title">📊 Case Intelligence</div>',
            unsafe_allow_html=True)

total_entities = len(risk_df)
high_risk = int((risk_df["risk_level"] == "High").sum()) if not risk_df.empty else 0
medium_risk = int((risk_df["risk_level"] == "Medium").sum()) if not risk_df.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    metric_card("Files Processed", analysis["files_processed"], "Evidence sources")

with m2:
    metric_card("Entities Found", total_entities, "Accounts / phones / IPs / devices")

with m3:
    metric_card("Relationships", len(edges), "Cross-entity links")

with m4:
    metric_card("High Risk", high_risk, "Immediate review")

with m5:
    metric_card("Medium Risk", medium_risk, "Requires triage")


# ---------------------- Risk overview ------------------------

st.markdown('<div class="section-title">🚨 Risk Intelligence</div>',
            unsafe_allow_html=True)

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Suspicious Entity Table")

    if risk_df.empty:
        st.warning("No correlated entities were found.")
    else:
        display_df = make_risk_table(risk_df)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Risk Distribution")

    if not risk_df.empty:
        distribution = (
            risk_df["risk_level"]
            .value_counts()
            .reindex(["High", "Medium", "Low"])
            .fillna(0)
            .astype(int)
        )

        st.bar_chart(distribution)

        st.markdown(
            f"""
            <div style="margin-top:12px;">
                <div>🔴 High: <b>{distribution.get("High", 0)}</b></div>
                <div>🟡 Medium: <b>{distribution.get("Medium", 0)}</b></div>
                <div>🟢 Low: <b>{distribution.get("Low", 0)}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------- Network -----------------------------

if show_graph:
    st.markdown('<div class="section-title">🕸️ Fraud Network Map</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
        <div class="small-mono">
        NETWORK VIEW :: ENTITY CORRELATION :: RISK-AWARE GRAPH
        </div>
        <p style="color:#94a3b8;">
        Follow linked accounts, UPI IDs, phones, devices and IP addresses.
        High-risk entities are emphasized by the backend graph renderer.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if os.path.exists(analysis["graph_path"]):
        st.image(
            analysis["graph_path"],
            use_container_width=True,
            caption="FRAUD-LINK AI entity correlation graph",
        )
    else:
        st.warning("Network graph could not be rendered.")


# ---------------------- Investigation tabs ------------------

st.markdown('<div class="section-title">🔬 Investigation Workspace</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Priority Entities",
    "🔗 Relationships",
    "📁 Evidence Preview",
    "🧠 Model Details",
])


with tab1:
    if risk_df.empty:
        st.info("No priority entities available.")
    else:
        priority = risk_df.sort_values(
            "risk_score", ascending=False
        ).head(10).copy()

        for _, row in priority.iterrows():
            level = row.get("risk_level", "Low")
            score = float(row.get("risk_score", 0))

            st.markdown(
                f"""
                <div class="panel" style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;">
                        <div>
                            <b>{row.get("entity", "Unknown")}</b>
                            <div class="small-mono">{row.get("entity_type", "unknown")}</div>
                        </div>
                        <div style="text-align:right;">
                            {risk_badge(level)}
                            <div style="font-size:22px;font-weight:700;">
                                {score:.1f}/100
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab2:
    if edges.empty:
        st.info("No relationships found.")
    else:
        edge_df = pd.DataFrame(edges)
        st.dataframe(
            edge_df,
            use_container_width=True,
            hide_index=True,
        )


with tab3:
    if not show_raw:
        st.info("Enable 'Show processed data' from the sidebar.")
    else:
        for name, df in analysis["processed_preview"].items():
            st.markdown(f"#### {name.replace('_', ' ').title()}")
            st.dataframe(
                df.head(100),
                use_container_width=True,
                hide_index=True,
            )


with tab4:
    st.write("### Decision Tree Risk Engine")

    report = analysis["model_report"]

    if isinstance(report, dict):
        st.json(report)
    else:
        st.write(report)

    training_source = report.get("training_source") if isinstance(report, dict) else None
    if training_source == "synthetic_demo":
        st.warning(
            "This is a demonstration model trained on synthetic labels. "
            "Upload labeled investigation data in the sidebar to retrain it."
        )
    elif training_source:
        st.success(f"Model source: {training_source}")


# -------------------- Evidence Integrity ---------------------

st.markdown('<div class="section-title">🔐 Evidence Integrity</div>',
            unsafe_allow_html=True)

st.markdown(
    """
    <div class="panel">
        <b>SHA-256 Preservation Layer</b>
        <p style="color:#94a3b8;margin-bottom:0;">
        Uploaded evidence is registered and hashed by the backend before
        processing. The hash can be used to verify that the source file
        remains unchanged during the analysis workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Export report ------------------------

st.markdown('<div class="section-title">📄 Investigative Brief</div>',
            unsafe_allow_html=True)

report_json = create_json_report(analysis)

download_col1, download_col2 = st.columns([1, 2])

with download_col1:
    st.download_button(
        "⬇️ Download JSON Brief",
        data=report_json,
        file_name="fraud_link_investigative_brief.json",
        mime="application/json",
        use_container_width=True,
    )

with download_col2:
    st.caption(
        "The JSON brief contains case summary and entity-level risk information "
        "generated from the current analysis."
    )


# -------------------------- Footer ---------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#64748b;font-size:11px;">
        FRAUD-LINK AI • Cyber Fraud Analysis & Digital Evidence Correlation
        <br>
        Built for rapid, explainable investigation triage
    </div>
    """,
    unsafe_allow_html=True,
)
