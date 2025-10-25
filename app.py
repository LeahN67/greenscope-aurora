import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import io
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image as PILImage, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Enhanced Configuration
# ------------------------------
class Config:
    PROJECT_TYPES = ["Forestry", "Renewable Energy", "Cookstoves", "Agriculture", "Waste Management", "Methane Capture"]
    COMPLIANCE_STANDARDS = ["Verra VCS", "Gold Standard", "Climate Action Reserve", "American Carbon Registry"]
    VERIFICATION_STATUSES = ["Verified", "Under Review", "Pending Verification", "Requires Correction"]
    REGIONS_COORDS = {
        "East Africa": [(-0.0236, 37.9062, "Kenya - Aberdare Forest"), (-6.7924, 39.2083, "Tanzania - Kilimanjaro"), (1.3733, 32.2903, "Uganda - Mabira Forest")],
        "West Africa": [(7.3775, -2.5367, "Ghana - Ashanti"), (9.0579, 8.6753, "Nigeria - Jos Plateau"), (12.2383, -1.5616, "Burkina Faso - Central")],
        "Southeast Asia": [(-0.7893, 113.9213, "Indonesia - Borneo"), (4.5353, 114.7277, "Brunei - Temburong"), (1.2966, 103.7764, "Singapore - Nature Reserves")],
        "South America": [(-3.4653, -62.2159, "Brazil - Amazon"), (-16.2902, -63.5887, "Bolivia - Santa Cruz"), (6.4238, -66.5897, "Venezuela - Orinoco Delta")],
    }
    TAGLINE = "Smart AI. Trusted Climate Impact."
    PRIMARY_COLOR = "#10b981"
    SECONDARY_COLOR = "#3b82f6"
    BACKGROUND_COLOR = "#f8fafc"
    CARD_BACKGROUND = "#ffffff"
    TEXT_COLOR = "#1e293b"
    TYPE_COLORS = {
        "Forestry": "#10b981",
        "Renewable Energy": "#f59e0b",
        "Cookstoves": "#06b6d4",
        "Agriculture": "#84cc16",
        "Waste Management": "#8b5cf6",
        "Methane Capture": "#3b82f6"
    }
    RISK_COLORS = {
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }
    SDG_MAPPING = {
        "Forestry": [13, 15, 8],
        "Cookstoves": [13, 5, 7, 3],
        "Renewable Energy": [7, 13, 8, 9],
        "Agriculture": [2, 13, 15, 5],
        "Waste Management": [11, 12, 13, 6],
        "Methane Capture": [13, 7, 9, 11]
    }
    SDG_TITLES = {
        1: "No Poverty", 2: "Zero Hunger", 3: "Good Health", 4: "Quality Education",
        5: "Gender Equality", 6: "Clean Water", 7: "Clean Energy", 8: "Decent Work",
        9: "Industry & Innovation", 10: "Reduced Inequalities", 11: "Sustainable Cities",
        12: "Responsible Consumption", 13: "Climate Action", 14: "Life Below Water",
        15: "Life on Land", 16: "Peace & Justice", 17: "Partnerships"
    }

# ------------------------------
# Custom CSS for Modern UI
# ------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        /* ... (your existing CSS remains unchanged) ... */
        </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Helper: SDG Badges
# ------------------------------
def render_sdg_badges(project_type: str):
    sdg_nums = Config.SDG_MAPPING.get(project_type, [])
    if not sdg_nums:
        st.caption("No SDG mapping available")
        return
    badge_html = ""
    for num in sdg_nums:
        title = Config.SDG_TITLES.get(num, f"SDG {num}")
        badge_html += f"""
        <span style="
            display: inline-block;
            background: #006699;
            color: white;
            padding: 0.25rem 0.5rem;
            margin: 0.25rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        ">SDG {num}<br><small>{title}</small></span>
        """
    st.markdown(badge_html, unsafe_allow_html=True)

# ------------------------------
# Scoring Functions
# ------------------------------
def calculate_gender_equity_score(row):
    score = 0
    if row.get('WomenLed', False):
        score += 40
    elif row.get('JobsCreated', 0) > 0:
        score += (row['WomenJobs'] / row['JobsCreated']) * 20
    income_generated = row.get('IncomeGenerated', 0)
    if income_generated > 0:
        score += (row['WomenIncomeShare'] / income_generated) * 30
    community_size = row.get('LocalCommunitySize', 1)
    if community_size > 0:
        score += min((row['WomenBeneficiaries'] / community_size) * 100, 20)
    training_provided = row.get('TrainingProvided', 0)
    if training_provided > 0:
        score += (row['WomenTrained'] / training_provided) * 10
    return min(max(score, 0), 100)

def calculate_youth_impact_score(row):
    score = 0
    if row.get('YouthInvolved', False):
        score += 30
    jobs_created = row.get('JobsCreated', 0)
    if jobs_created > 0:
        score += (row['YouthJobs'] / jobs_created) * 25
    community_size = row.get('LocalCommunitySize', 1)
    if community_size > 0:
        score += min((row['YouthBeneficiaries'] / community_size) * 100, 25)
    training_provided = row.get('TrainingProvided', 0)
    if training_provided > 0:
        score += (row['YouthTrained'] / training_provided) * 20
    return min(max(score, 0), 100)

# ------------------------------
# Generate Impact Data
# ------------------------------
def generate_impact_data(df):
    df = df.copy()
    df['WomenLed'] = np.random.choice([True, False], len(df), p=[0.35, 0.65])
    df['YouthInvolved'] = np.random.choice([True, False], len(df), p=[0.42, 0.58])
    df['WomenBeneficiaries'] = np.random.randint(50, 5000, len(df))
    df['YouthBeneficiaries'] = np.random.randint(30, 3000, len(df))
    df['JobsCreated'] = np.random.randint(10, 500, len(df))
    df['WomenJobs'] = (df['JobsCreated'] * np.random.uniform(0.3, 0.7, len(df))).astype(int)
    df['YouthJobs'] = (df['JobsCreated'] * np.random.uniform(0.25, 0.6, len(df))).astype(int)
    df['LocalCommunitySize'] = np.random.randint(500, 50000, len(df))
    df['HouseholdsImpacted'] = np.random.randint(100, 10000, len(df))
    df['WomenHeadedHouseholds'] = (df['HouseholdsImpacted'] * np.random.uniform(0.2, 0.5, len(df))).astype(int)
    df['IncomeGenerated'] = df['MarketValue'] * np.random.uniform(0.05, 0.25, len(df))
    df['WomenIncomeShare'] = df['IncomeGenerated'] * np.random.uniform(0.3, 0.6, len(df))
    df['TrainingProvided'] = np.random.randint(20, 500, len(df))
    df['WomenTrained'] = (df['TrainingProvided'] * np.random.uniform(0.4, 0.7, len(df))).astype(int)
    df['YouthTrained'] = (df['TrainingProvided'] * np.random.uniform(0.3, 0.6, len(df))).astype(int)
    df['GenderEquityScore'] = df.apply(calculate_gender_equity_score, axis=1)
    df['YouthImpactScore'] = df.apply(calculate_youth_impact_score, axis=1)
    return df

# ------------------------------
# Enhanced Data Management (unchanged)
# ------------------------------
class CarbonProjectDataManager:
    # ... (keep your existing implementation exactly as is) ...
    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_realistic_carbon_projects(n_projects: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is None:
            seed = int(datetime.now().timestamp()) % 10000
        np.random.seed(seed)
        data = []
        for i in range(n_projects):
            project_id = f"VCS-{2020 + (i % 5)}-{1000+i}"
            project_type = np.random.choice(Config.PROJECT_TYPES)
            region = np.random.choice(list(Config.REGIONS_COORDS.keys()))
            lat, lon, location_name = Config.REGIONS_COORDS[region][np.random.randint(0, len(Config.REGIONS_COORDS[region]))]
            lat += np.random.uniform(-0.5, 0.5)
            lon += np.random.uniform(-0.5, 0.5)
            baseline_emissions = np.random.randint(5000, 50000)
            reduction_rate = np.random.uniform(0.15, 0.85)
            credits_generated = int(baseline_emissions * reduction_rate)
            buffer_percentage = np.random.uniform(0.10, 0.20)
            buffer_credits = int(credits_generated * buffer_percentage)
            net_credits = credits_generated - buffer_credits
            permanence_risk = np.random.choice(["Low", "Medium", "High"], p=[0.5, 0.35, 0.15])
            additionality_score = np.random.uniform(65, 98)
            verification_status = np.random.choice(Config.VERIFICATION_STATUSES, p=[0.6, 0.2, 0.15, 0.05])
            compliance_standard = np.random.choice(Config.COMPLIANCE_STANDARDS)
            start_date = datetime.now() - timedelta(days=np.random.randint(365, 365*5))
            vintage_year = start_date.year + np.random.randint(0, 5)
            co2_price = np.random.uniform(15, 85)
            market_value = net_credits * co2_price
            data.append([
                project_id, project_type, region, location_name,
                credits_generated, net_credits, buffer_credits, baseline_emissions,
                np.random.randint(1, 10), verification_status, compliance_standard,
                permanence_risk, additionality_score, vintage_year,
                start_date, datetime.now(), lat, lon, co2_price, market_value
            ])
        columns = [
            "ProjectID", "Type", "Region", "Location", 
            "TotalCredits", "NetCredits", "BufferCredits", "BaselineEmissions",
            "MonitoringPeriod", "VerificationStatus", "ComplianceStandard",
            "PermanenceRisk", "AdditionalityScore", "VintageYear",
            "StartDate", "LastUpdate", "Latitude", "Longitude", "CO2Price", "MarketValue"
        ]
        return pd.DataFrame(data, columns=columns)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        issues = []
        warnings = []
        if df.empty:
            issues.append("DataFrame is empty")
            return {"valid": False, "issues": issues, "warnings": warnings}
        if not df['NetCredits'].ge(0).all():
            issues.append("Negative NetCredits detected")
        if not df['AdditionalityScore'].between(0, 100).all():
            issues.append("AdditionalityScore outside valid range [0,100]")
        if not (df['BufferCredits'] <= df['TotalCredits']).all():
            issues.append("BufferCredits exceed TotalCredits")
        if df[['Latitude', 'Longitude']].isnull().any().any():
            issues.append("Missing geographic coordinates")
        if df['NetCredits'].median() == 0:
            warnings.append("Median NetCredits is zero")
        if (df['AdditionalityScore'] < 70).sum() > len(df) * 0.3:
            warnings.append("Over 30% of projects have low additionality scores (<70)")
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_projects": len(df),
            "verified_count": len(df[df['VerificationStatus'] == 'Verified'])
        }

# ------------------------------
# Enhanced Filtering (BASE ONLY)
# ------------------------------
@st.cache_data
def apply_filters(
    df: pd.DataFrame,
    regions: list,
    types: list,
    statuses: list,
    risks: list,
    credit_range: Tuple[int, int],
    additionality_range: Tuple[float, float],
    vintage_range: Tuple[int, int]
) -> pd.DataFrame:
    filtered = df.copy()
    if regions:
        filtered = filtered[filtered["Region"].isin(regions)]
    if types:
        filtered = filtered[filtered["Type"].isin(types)]
    if statuses:
        filtered = filtered[filtered["VerificationStatus"].isin(statuses)]
    if risks:
        filtered = filtered[filtered["PermanenceRisk"].isin(risks)]
    filtered = filtered[
        filtered["NetCredits"].between(credit_range[0], credit_range[1]) &
        filtered["AdditionalityScore"].between(additionality_range[0], additionality_range[1]) &
        filtered["VintageYear"].between(vintage_range[0], vintage_range[1])
    ]
    return filtered

# ------------------------------
# Social Impact Tab
# ------------------------------
def render_impact_tab(df):
    if df.empty:
        st.info("No projects to display in Social Impact dashboard.")
        return
    st.markdown("### 🌍 Social Impact Dashboard")
    st.markdown("**Measuring climate action through an equity lens**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        women_led = df['WomenLed'].sum()
        women_led_pct = (women_led / len(df)) * 100
        st.metric("👩‍💼 Women-Led Projects", f"{women_led:,}", f"{women_led_pct:.1f}%")
    with col2:
        st.metric("👩 Women Beneficiaries", f"{df['WomenBeneficiaries'].sum():,.0f}")
    with col3:
        youth_projects = df['YouthInvolved'].sum()
        st.metric("🎓 Youth-Involved Projects", f"{youth_projects:,}", f"{(youth_projects/len(df)*100):.1f}%")
    with col4:
        st.metric("💼 Jobs for Women", f"{df['WomenJobs'].sum():,.0f}")
    with col5:
        st.metric("📚 Youth Trained", f"{df['YouthTrained'].sum():,.0f}")
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Equity Scores")
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        avg_gender = df['GenderEquityScore'].mean()
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #ec4899;">
            <h4 style="color: #0f172a; margin-top: 0;">Gender Equity Score</h4>
            <p style="font-size: 2.5rem; font-weight: 800; color: #ec4899; margin: 0.5rem 0;">{avg_gender:.1f}/100</p>
            <p style="color: #64748b; margin: 0;">Leadership, income, participation & training</p>
        </div>
        """, unsafe_allow_html=True)
    with score_col2:
        avg_youth = df['YouthImpactScore'].mean()
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #8b5cf6;">
            <h4 style="color: #0f172a; margin-top: 0;">Youth Impact Score</h4>
            <p style="font-size: 2.5rem; font-weight: 800; color: #8b5cf6; margin: 0.5rem 0;">{avg_youth:.1f}/100</p>
            <p style="color: #64748b; margin: 0;">Involvement, jobs, reach & skills</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    # SDG Section
    all_sdgs = set()
    for ptype in df['Type'].unique():
        all_sdgs.update(Config.SDG_MAPPING.get(ptype, []))
    st.markdown("### 🌐 UN Sustainable Development Goals Alignment")
    st.info(f"This portfolio supports **{len(all_sdgs)} of the 17 UN SDGs**, with strong focus on Climate Action (SDG 13), Gender Equality (SDG 5), and Decent Work (SDG 8).")
    # Export
    st.markdown("### 📥 Export Social Impact Data")
    impact_summary = pd.DataFrame({
        'Metric': ['Total Women Beneficiaries', 'Total Youth Beneficiaries', 'Women-Led Projects', 'Youth-Involved Projects', 'Jobs for Women', 'Jobs for Youth', 'Women Trained', 'Youth Trained', 'Income to Women (USD)', 'Women-Headed Households Supported', 'Avg Gender Equity Score', 'Avg Youth Impact Score'],
        'Value': [
            df['WomenBeneficiaries'].sum(),
            df['YouthBeneficiaries'].sum(),
            df['WomenLed'].sum(),
            df['YouthInvolved'].sum(),
            df['WomenJobs'].sum(),
            df['YouthJobs'].sum(),
            df['WomenTrained'].sum(),
            df['YouthTrained'].sum(),
            df['WomenIncomeShare'].sum(),
            df['WomenHeadedHouseholds'].sum(),
            round(df['GenderEquityScore'].mean(), 1),
            round(df['YouthImpactScore'].mean(), 1)
        ]
    })
    csv = impact_summary.to_csv(index=False).encode('utf-8')
    st.download_button("📊 Download Impact Summary (CSV)", data=csv, file_name=f"social_impact_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)

# ------------------------------
# Enhanced Visualization Manager (updated popup)
# ------------------------------
class VisualizationManager:
    @staticmethod
    def _apply_modern_theme(fig):
        fig.update_layout(
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1e293b"),
            title=dict(font=dict(size=18, color="#0f172a", family="Inter"), x=0.5, xanchor='center'),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False, showline=True, linecolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False, showline=True, linecolor="#e2e8f0"),
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter")
        )
        return fig

    @staticmethod
    def create_enhanced_popup(row) -> str:
        status_color = "#10b981" if row['VerificationStatus'] == 'Verified' else "#f59e0b"
        risk_color = Config.RISK_COLORS.get(row['PermanenceRisk'], "#64748b")
        sdgs = Config.SDG_MAPPING.get(row['Type'], [])
        sdg_html = ", ".join([f"SDG {s}" for s in sdgs]) if sdgs else "Not mapped"
        return f"""
        <div style="font-family: Inter, sans-serif; width: 280px;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 12px; border-radius: 8px 8px 0 0; margin: -10px -10px 10px -10px;">
                <h4 style="margin: 0; font-size: 16px;">{row['ProjectID']}</h4>
                <p style="margin: 4px 0 0 0; font-size: 12px; opacity: 0.9;">{row['Type']}</p>
            </div>
            <div style="padding: 0 4px;">
                <p style="margin: 8px 0; font-size: 13px;">
                    <strong>📍 Location:</strong> {row['Location']}<br>
                    <strong>💰 Net Credits:</strong> {row['NetCredits']:,.0f} tCO₂e<br>
                    <strong>💵 Market Value:</strong> ${row['MarketValue']:,.0f}<br>
                    <strong>📅 Vintage:</strong> {row['VintageYear']}<br>
                    <strong>✅ Status:</strong> <span style="color: {status_color}; font-weight: 600;">{row['VerificationStatus']}</span><br>
                    <strong>⚠️ Risk:</strong> <span style="color: {risk_color}; font-weight: 600;">{row['PermanenceRisk']}</span><br>
                    <strong>📊 Additionality:</strong> {row['AdditionalityScore']:.1f}%<br>
                    <strong>🌍 SDGs:</strong> {sdg_html}
                </p>
            </div>
        </div>
        """

    # ... (other methods unchanged) ...

# ------------------------------
# Report Generator (unchanged)
# ------------------------------
class ReportGenerator:
    # ... (keep your existing implementation) ...
    pass

# ------------------------------
# Render Functions (updated)
# ------------------------------
def render_header():
    st.markdown("""
        <div class="main-header">
            <div class="main-title">🌍 GreenScope Analytics</div>
            <div class="main-subtitle">Smart AI. Trusted Climate Impact.</div>
        </div>
    """, unsafe_allow_html=True)

def render_compare_tab(df: pd.DataFrame):
    st.markdown("### ⚖️ Project Comparison")
    project_ids = sorted(df["ProjectID"].unique())
    if len(project_ids) < 2:
        st.warning("Need at least 2 projects to compare")
        return
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Select First Project", project_ids, key="compare_p1")
    with col2:
        available_projects = [p for p in project_ids if p != p1]
        p2 = st.selectbox("Select Second Project", available_projects, key="compare_p2")
    if p1 and p2 and p1 != p2:
        d1 = df[df["ProjectID"] == p1].iloc[0]
        d2 = df[df["ProjectID"] == p2].iloc[0]
        card_col1, card_col2 = st.columns(2)
        with card_col1:
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #10b981;">
                    <h3 style="color: #0f172a; margin-top: 0;">{d1['ProjectID']}</h3>
                    <p><strong>Type:</strong> {d1['Type']}</p>
                    <p><strong>Location:</strong> {d1['Location']}</p>
                    <p><strong>Region:</strong> {d1['Region']}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("#### Metrics")
            st.metric("Net Credits", f"{d1['NetCredits']:,.0f} tCO₂e")
            st.metric("Market Value", f"${d1['MarketValue']:,.0f}")
            st.metric("Additionality Score", f"{d1['AdditionalityScore']:.1f}%")
            st.metric("CO₂ Price", f"${d1['CO2Price']:.2f}")
            st.markdown("#### Status")
            st.write(f"**Verification:** {d1['VerificationStatus']}")
            st.write(f"**Risk Level:** {d1['PermanenceRisk']}")
            st.write(f"**Standard:** {d1['ComplianceStandard']}")
            st.write(f"**Vintage:** {d1['VintageYear']}")
            st.markdown("**Aligned with UN SDGs:**")
            render_sdg_badges(d1['Type'])
        with card_col2:
            st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6;">
                    <h3 style="color: #0f172a; margin-top: 0;">{d2['ProjectID']}</h3>
                    <p><strong>Type:</strong> {d2['Type']}</p>
                    <p><strong>Location:</strong> {d2['Location']}</p>
                    <p><strong>Region:</strong> {d2['Region']}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("#### Metrics")
            delta_credits = d2['NetCredits'] - d1['NetCredits']
            st.metric("Net Credits", f"{d2['NetCredits']:,.0f} tCO₂e", delta=f"{delta_credits:+,.0f}")
            delta_value = d2['MarketValue'] - d1['MarketValue']
            st.metric("Market Value", f"${d2['MarketValue']:,.0f}", delta=f"${delta_value:+,.0f}")
            delta_add = d2['AdditionalityScore'] - d1['AdditionalityScore']
            st.metric("Additionality Score", f"{d2['AdditionalityScore']:.1f}%", delta=f"{delta_add:+.1f}%")
            delta_price = d2['CO2Price'] - d1['CO2Price']
            st.metric("CO₂ Price", f"${d2['CO2Price']:.2f}", delta=f"${delta_price:+.2f}")
            st.markdown("#### Status")
            st.write(f"**Verification:** {d2['VerificationStatus']}")
            st.write(f"**Risk Level:** {d2['PermanenceRisk']}")
            st.write(f"**Standard:** {d2['ComplianceStandard']}")
            st.write(f"**Vintage:** {d2['VintageYear']}")
            st.markdown("**Aligned with UN SDGs:**")
            render_sdg_badges(d2['Type'])

        # ... (rest of comparison chart unchanged) ...

def render_advanced_filters():
    st.sidebar.markdown("## 🔍 Advanced Filters")
    with st.sidebar.expander("📍 Geographic Filters", expanded=True):
        regions = st.multiselect("Regions", sorted(st.session_state.df["Region"].unique()), default=sorted(st.session_state.df["Region"].unique()), key="region_filter")
    with st.sidebar.expander("🏷️ Project Filters", expanded=True):
        types = st.multiselect("Project Types", sorted(st.session_state.df["Type"].unique()), default=sorted(st.session_state.df["Type"].unique()), key="type_filter")
        statuses = st.multiselect("Verification Status", Config.VERIFICATION_STATUSES, default=Config.VERIFICATION_STATUSES, key="status_filter")
        risks = st.multiselect("Permanence Risk", ["Low", "Medium", "High"], default=["Low", "Medium", "High"], key="risk_filter")
    with st.sidebar.expander("📊 Metric Filters", expanded=True):
        credit_range = st.slider("Net Credits (tCO₂e)", int(st.session_state.df["NetCredits"].min()), int(st.session_state.df["NetCredits"].max()), (int(st.session_state.df["NetCredits"].min()), int(st.session_state.df["NetCredits"].max())), key="credit_range")
        additionality_range = st.slider("Additionality Score (%)", float(st.session_state.df["AdditionalityScore"].min()), 100.0, (float(st.session_state.df["AdditionalityScore"].min()), 100.0), key="additionality_range")
        vintage_range = st.slider("Vintage Year", int(st.session_state.df["VintageYear"].min()), int(st.session_state.df["VintageYear"].max()), (int(st.session_state.df["VintageYear"].min()), int(st.session_state.df["VintageYear"].max())), key="vintage_range")
    with st.sidebar.expander("🌍 Social Impact Filters", expanded=True):
        show_women_led = st.checkbox("Show Only Women-Led Projects", value=False)
        show_youth = st.checkbox("Show Only Youth-Involved Projects", value=False)
        min_women_jobs = st.slider("Minimum Jobs for Women", 0, 500, 0)
        min_women_beneficiaries = st.slider("Minimum Women Beneficiaries", 0, 5000, 0)
    with st.sidebar.expander("🤖 AI Equity Filters", expanded=False):
        min_gender_score = st.slider("Minimum Gender Equity Score", 0, 100, 0)
        min_youth_score = st.slider("Minimum Youth Impact Score", 0, 100, 0)
    return (
        regions, types, statuses, risks,
        credit_range, additionality_range, vintage_range,
        show_women_led, show_youth, min_women_jobs, min_women_beneficiaries,
        min_gender_score, min_youth_score
    )

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.set_page_config(
        page_title="GreenScope Analytics | Climate Intelligence Platform",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    render_header()
    st.sidebar.markdown("## ⚙️ Configuration")
    with st.sidebar.expander("🔧 Data Settings", expanded=False):
        fixed_dataset = st.checkbox("Use Fixed Dataset (seed=42)", value=True)
        n_projects = st.slider("Number of Projects", 50, 200, 100)
        if st.button("🔄 Regenerate Data"):
            st.session_state.pop('df', None)
            st.rerun()
    if 'df' not in st.session_state:
        with st.spinner("🔄 Loading portfolio data..."):
            seed = 42 if fixed_dataset else None
            st.session_state.df = CarbonProjectDataManager.generate_realistic_carbon_projects(n_projects=n_projects, seed=seed)
    df = st.session_state.df
    validation = CarbonProjectDataManager.validate_dataframe(df)
    if not validation["valid"]:
        st.error("⚠️ Data Validation Issues Detected:")
        for issue in validation["issues"]:
            st.error(f"- {issue}")
        return
    if validation.get("warnings"):
        with st.expander("⚠️ Data Warnings", expanded=False):
            for warning in validation["warnings"]:
                st.warning(warning)
    # Enhance with impact data
    if 'df_with_impact' not in st.session_state:
        st.session_state.df_with_impact = generate_impact_data(st.session_state.df.copy())
    df_with_impact = st.session_state.df_with_impact
    # Get filters
    (
        regions, types, statuses, risks,
        credit_range, additionality_range, vintage_range,
        show_women_led, show_youth, min_women_jobs, min_women_beneficiaries,
        min_gender_score, min_youth_score
    ) = render_advanced_filters()
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if 'filter' in key or 'range' in key:
                del st.session_state[key]
        st.rerun()
    # Apply BASE filters
    filtered_df = apply_filters(
        df, regions, types, statuses, risks,
        credit_range, additionality_range, vintage_range
    )
    # Apply impact filters
    impact_filtered = df_with_impact[df_with_impact.index.isin(filtered_df.index)]
    if show_women_led:
        impact_filtered = impact_filtered[impact_filtered['WomenLed']]
    if show_youth:
        impact_filtered = impact_filtered[impact_filtered['YouthInvolved']]
    impact_filtered = impact_filtered[
        (impact_filtered['WomenJobs'] >= min_women_jobs) &
        (impact_filtered['WomenBeneficiaries'] >= min_women_beneficiaries) &
        (impact_filtered['GenderEquityScore'] >= min_gender_score) &
        (impact_filtered['YouthImpactScore'] >= min_youth_score)
    ]
    final_filtered_df = df.loc[impact_filtered.index]
    final_filtered_df_with_impact = impact_filtered
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📊 Filtered Results")
    st.sidebar.metric("Projects Shown", f"{len(final_filtered_df):,}")
    st.sidebar.metric("Total Credits", f"{final_filtered_df['NetCredits'].sum():,.0f}")
    if len(final_filtered_df) == 0:
        st.warning("⚠️ No projects match your current filters. Please adjust your criteria.")
        return
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard",
        "📈 Analytics",
        "🗺️ Geographic Map",
        "⚖️ Compare Projects",
        "📋 Reports",
        "📑 Data Table",
        "🌍 Social Impact"
    ])
    with tab1:
        render_dashboard_tab(final_filtered_df)
    with tab2:
        render_analytics_tab(final_filtered_df)
    with tab3:
        render_map_tab(final_filtered_df)
    with tab4:
        render_compare_tab(final_filtered_df)
    with tab5:
        render_report_tab(final_filtered_df)
    with tab6:
        st.markdown("### 📑 Project Data Table")
        all_columns = final_filtered_df.columns.tolist()
        default_columns = ['ProjectID', 'Type', 'Region', 'NetCredits', 'VerificationStatus', 'PermanenceRisk', 'AdditionalityScore', 'MarketValue']
        selected_columns = st.multiselect("Select Columns to Display", all_columns, default=[col for col in default_columns if col in all_columns])
        if selected_columns:
            search_term = st.text_input("🔍 Search Projects", "")
            display_df = final_filtered_df[selected_columns].copy()
            if search_term:
                mask = display_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)
                display_df = display_df[mask]
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by = st.selectbox("Sort By", selected_columns)
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            display_df = display_df.sort_values(by=sort_by, ascending=(sort_order == "Ascending"))
            format_dict = {}
            format_specs = {
                'NetCredits': '{:,.0f}', 'TotalCredits': '{:,.0f}', 'BufferCredits': '{:,.0f}',
                'BaselineEmissions': '{:,.0f}', 'AdditionalityScore': '{:.1f}%',
                'MarketValue': '${:,.0f}', 'CO2Price': '${:.2f}'
            }
            for col_name, fmt in format_specs.items():
                if col_name in display_df.columns:
                    format_dict[col_name] = fmt
            st.dataframe(display_df.style.format(format_dict), use_container_width=True, height=500)
            st.caption(f"Showing {len(display_df)} of {len(final_filtered_df)} projects")
        else:
            st.info("Please select at least one column to display")
    with tab7:
        render_impact_tab(final_filtered_df_with_impact)
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div class="footer">
            <p><strong>GreenScope Analytics</strong> — Climate Intelligence Platform</p>
            <p>Showing {len(final_filtered_df):,} of {len(df):,} projects | 
               Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | 
               Total Portfolio Value: ${final_filtered_df['MarketValue'].sum():,.0f}</p>
            <p style="font-size: 0.75rem; margin-top: 1rem;">
                © {datetime.now().year} GreenScope Analytics. All rights reserved. | 
                <a href="#" style="color: #10b981; text-decoration: none;">Privacy Policy</a> | 
                <a href="#" style="color: #10b981; text-decoration: none;">Terms of Service</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# Keep other render functions (dashboard, analytics, map, report) as in your original code
# For brevity, they are not repeated here but must be present in your file

if __name__ == "__main__":
    main()