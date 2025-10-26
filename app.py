import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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


# ------------------------------
# Custom CSS for Modern UI
# ------------------------------
def inject_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f8fafc;
        }
        
        .main-header {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .main-title {
            color: white;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .main-subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.2rem;
            margin-top: 0.5rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: white;
            padding: 0.5rem;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            color: #64748b;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #10b981;
            color: white;
        }
        
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: #64748b;
            font-size: 0.875rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ------------------------------
# Enhanced Data Management
# ------------------------------
class CarbonProjectDataManager:
    @staticmethod
    @st.cache_data(ttl=3600)
    def generate_realistic_carbon_projects(n_projects: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate realistic carbon project data with caching"""
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
        """Enhanced data validation"""
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
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_projects": len(df),
            "verified_count": len(df[df['VerificationStatus'] == 'Verified'])
        }


# ------------------------------
# Social Impact Data Generation
# ------------------------------
def generate_impact_data(df):
    """Add social impact metrics to existing project data"""
    np.random.seed(42)
    
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
    
    return df


# ------------------------------
# AI Equity Intelligence System
# ------------------------------
class EquityIntelligenceAI:
    """AI system that detects inconsistencies in social impact claims"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.15, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_social_impact_greenwashing(self, df):
        """Use AI to flag projects with suspicious social impact claims"""
        features = df[[
            'WomenBeneficiaries', 'YouthBeneficiaries', 
            'WomenJobs', 'YouthJobs',
            'WomenIncomeShare', 'NetCredits',
            'AdditionalityScore', 'LocalCommunitySize'
        ]].copy()
        
        features_scaled = self.scaler.fit_transform(features)
        df['EquityAnomalyScore'] = self.anomaly_detector.fit_predict(features_scaled)
        df['IsSociallyFlagged'] = df['EquityAnomalyScore'] == -1
        df['RedFlags'] = df.apply(self._identify_red_flags, axis=1)
        
        return df
    
    def _identify_red_flags(self, row):
        """Identify specific equity red flags"""
        flags = []
        
        if row['NetCredits'] > 20000 and row['WomenBeneficiaries'] < 100:
            flags.append("⚠️ High carbon credits but very few women beneficiaries")
        
        if row['WomenLed'] and row['WomenJobs'] / max(row['JobsCreated'], 1) < 0.3:
            flags.append("⚠️ Claims women-led but low women employment")
        
        if row['YouthInvolved'] and row['YouthJobs'] < 5:
            flags.append("⚠️ Claims youth involvement but minimal youth employment")
        
        income_ratio = row['IncomeGenerated'] / max(row['MarketValue'], 1)
        if income_ratio < 0.05:
            flags.append("⚠️ Less than 5% of revenue reaches local communities")
        
        if row['LocalCommunitySize'] > 10000 and row['WomenBeneficiaries'] + row['YouthBeneficiaries'] < 200:
            flags.append("⚠️ Large community but minimal direct impact")
        
        if row['TrainingProvided'] > 100 and row['WomenTrained'] / max(row['TrainingProvided'], 1) < 0.2:
            flags.append("⚠️ Low women participation in training programs")
        
        return flags if flags else []
    
    def calculate_equity_excellence_score(self, df):
        """AI-generated composite score: combines carbon impact with social equity"""
        scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Carbon Impact (40 points)
            carbon_score = min((row['NetCredits'] / df['NetCredits'].quantile(0.95)) * 40, 40)
            score += carbon_score
            
            # Gender Equity (30 points)
            gender_score = 0
            if row['WomenLed']:
                gender_score += 12
            women_job_ratio = row['WomenJobs'] / max(row['JobsCreated'], 1)
            gender_score += women_job_ratio * 10
            women_income_ratio = row['WomenIncomeShare'] / max(row['IncomeGenerated'], 1)
            gender_score += women_income_ratio * 8
            score += min(gender_score, 30)
            
            # Youth Empowerment (20 points)
            youth_score = 0
            if row['YouthInvolved']:
                youth_score += 8
            youth_job_ratio = row['YouthJobs'] / max(row['JobsCreated'], 1)
            youth_score += youth_job_ratio * 7
            youth_training_ratio = row['YouthTrained'] / max(row['TrainingProvided'], 1)
            youth_score += youth_training_ratio * 5
            score += min(youth_score, 20)
            
            # Community Reach (10 points)
            beneficiary_ratio = (row['WomenBeneficiaries'] + row['YouthBeneficiaries']) / max(row['LocalCommunitySize'], 1)
            community_score = min(beneficiary_ratio * 100, 10)
            score += community_score
            
            # Penalty for red flags
            if len(row['RedFlags']) > 0:
                score *= (1 - len(row['RedFlags']) * 0.1)
            
            scores.append(min(score, 100))
        
        df['EquityExcellenceScore'] = scores
        return df
    
    def predict_impact_potential(self, df):
        """Predict which projects have unrealized potential for greater social impact"""
        df['ImpactPotential'] = 'Standard'
        
        high_carbon = df['NetCredits'] > df['NetCredits'].quantile(0.75)
        low_equity = df['EquityExcellenceScore'] < 60
        df.loc[high_carbon & low_equity, 'ImpactPotential'] = 'High Potential - Needs Social Focus'
        
        high_equity = df['EquityExcellenceScore'] > 75
        mid_carbon = df['NetCredits'] < df['NetCredits'].quantile(0.75)
        df.loc[high_equity & mid_carbon, 'ImpactPotential'] = 'Equity Leader - Scale Carbon'
        
        balanced = (df['EquityExcellenceScore'] > 75) & (df['NetCredits'] > df['NetCredits'].quantile(0.75))
        df.loc[balanced, 'ImpactPotential'] = '⭐ Balanced Excellence'
        
        return df
    
    def identify_equity_champions(self, df, top_n=10):
        """AI identifies the highest-impact projects combining carbon + social equity"""
        df_sorted = df.sort_values('EquityExcellenceScore', ascending=False)
        champions = df_sorted.head(top_n)
        return champions


# ------------------------------
# Enhanced Filtering
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
    """Apply multiple filters with caching"""
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
# Enhanced Visualization Manager
# ------------------------------
class VisualizationManager:
    @staticmethod
    def _apply_modern_theme(fig):
        """Apply modern, clean theme to figures"""
        fig.update_layout(
            font=dict(family="Inter, system-ui, sans-serif", size=12, color="#1e293b"),
            title=dict(font=dict(size=18, color="#0f172a", family="Inter"), x=0.5, xanchor='center'),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, zeroline=False),
            margin=dict(l=60, r=40, t=80, b=60),
            hovermode="closest"
        )
        return fig

    @staticmethod
    def credits_by_type_enhanced(df):
        """Enhanced bar chart with better styling"""
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="#64748b")
            )
        
        agg_df = df.groupby(['Region', 'Type'])['NetCredits'].sum().reset_index()
        
        fig = px.bar(
            agg_df,
            x="Type",
            y="NetCredits",
            color="Type",
            facet_col="Region",
            facet_col_wrap=2,
            title="Net Carbon Credits by Project Type & Region",
            labels={"NetCredits": "Net Credits (tCO₂e)", "Type": "Project Type"},
            color_discrete_map=Config.TYPE_COLORS,
            height=600
        )
        
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def compliance_donut(df):
        """Modern donut chart for compliance standards"""
        if df.empty:
            return go.Figure()
        
        compliance_counts = df['ComplianceStandard'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=compliance_counts.index,
            values=compliance_counts.values,
            hole=0.5,
            marker=dict(colors=['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b']),
            textposition='outside',
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title="Compliance Standards Distribution",
            showlegend=False,
            annotations=[dict(text=f'{len(df)}<br>Projects', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def risk_bubble_chart(df, max_points=500):
        """Enhanced bubble chart with risk analysis"""
        if df.empty:
            return go.Figure()
        
        df_plot = df.copy()
        if len(df_plot) > max_points:
            df_plot = df_plot.sample(n=max_points, random_state=42)
        
        fig = px.scatter(
            df_plot,
            x="AdditionalityScore",
            y="NetCredits",
            size="MarketValue",
            color="PermanenceRisk",
            hover_data=["ProjectID", "Type", "Region"],
            title="Risk Profile: Additionality vs. Net Credits",
            labels={
                "AdditionalityScore": "Additionality Score (%)",
                "NetCredits": "Net Credits (tCO₂e)"
            },
            color_discrete_map=Config.RISK_COLORS,
            size_max=60
        )
        
        return VisualizationManager._apply_modern_theme(fig)

    @staticmethod
    def create_enhanced_popup(row) -> str:
        """Create beautiful popup for map markers"""
        status_color = "#10b981" if row['VerificationStatus'] == 'Verified' else "#f59e0b"
        risk_color = Config.RISK_COLORS.get(row['PermanenceRisk'], "#64748b")
        
        return f"""
        <div style="font-family: Inter, sans-serif; width: 280px;">
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 12px; border-radius: 8px 8px 0 0; margin: -10px -10px 10px -10px;">
                <h4 style="margin: 0; font-size: 16px;">{row['ProjectID']}</h4>
                <p style="margin: 4px 0 0 0; font-size: 12px; opacity: 0.9;">{row['Type']}</p>
            </div>
            <div style="padding: 0 4px;">
                <p style="margin: 8px 0; font-size: 13px;">
                    <strong>Location:</strong> {row['Location']}<br>
                    <strong>Net Credits:</strong> {row['NetCredits']:,.0f} tCO₂e<br>
                    <strong>Market Value:</strong> ${row['MarketValue']:,.0f}<br>
                    <strong>Status:</strong> <span style="color: {status_color};">{row['VerificationStatus']}</span><br>
                    <strong>Risk:</strong> <span style="color: {risk_color};">{row['PermanenceRisk']}</span>
                </p>
            </div>
        </div>
        """


# ------------------------------
# Enhanced Report Generator
# ------------------------------
class ReportGenerator:
    @staticmethod
    def generate_pdf(df: pd.DataFrame) -> Optional[bytes]:
        """Generate comprehensive PDF report"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                topMargin=0.6 * inch,
                bottomMargin=0.6 * inch,
                leftMargin=0.7 * inch,
                rightMargin=0.7 * inch
            )
            
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=26,
                textColor=rl_colors.Color(0.06, 0.73, 0.51),
                spaceAfter=8,
                alignment=1,
                fontName='Helvetica-Bold'
            )
            
            centered_style = ParagraphStyle(
                'Centered',
                parent=styles['Normal'],
                alignment=1,
                fontSize=12,
                leading=16
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=rl_colors.Color(0.06, 0.73, 0.51),
                spaceAfter=12,
                fontName='Helvetica-Bold'
            )
            
            story = []

            # Header
            story.append(Paragraph("GreenScope Analytics", title_style))
            story.append(Paragraph("<b>Smart AI. Trusted Climate Impact.</b>", centered_style))
            story.append(Spacer(1, 24))

            # Executive Summary
            story.append(Paragraph("<b>Executive Summary</b>", heading_style))
            story.append(Spacer(1, 8))
            
            total_market_value = df['MarketValue'].sum()
            avg_price = df['CO2Price'].mean()
            
            summary_data = [
                ["Metric", "Value"],
                ["Total Active Projects", f"{len(df):,}"],
                ["Total Net Credits Generated", f"{df['NetCredits'].sum():,.0f} tCO2e"],
                ["Total Market Value", f"${total_market_value:,.0f}"],
                ["Average CO2 Price", f"${avg_price:.2f}/tCO2e"],
                ["Average Additionality Score", f"{df['AdditionalityScore'].mean():.1f}%"],
                ["Verified Projects", f"{len(df[df['VerificationStatus']=='Verified'])} ({(len(df[df['VerificationStatus']=='Verified'])/len(df)*100):.1f}%)"],
            ]
            
            summary_table = Table(summary_data, colWidths=[3.2*inch, 2.2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.06, 0.73, 0.51)),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME', (1,1), (1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 11),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 1, rl_colors.grey),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 24))

            # Social Impact Summary
            if 'WomenBeneficiaries' in df.columns:
                story.append(Paragraph("<b>Social Impact Summary</b>", heading_style))
                story.append(Spacer(1, 8))
                
                impact_data = [
                    ["Impact Metric", "Value"],
                    ["Women Beneficiaries", f"{df['WomenBeneficiaries'].sum():,}"],
                    ["Youth Beneficiaries", f"{df['YouthBeneficiaries'].sum():,}"],
                    ["Jobs for Women", f"{df['WomenJobs'].sum():,}"],
                    ["Jobs for Youth", f"{df['YouthJobs'].sum():,}"],
                    ["Women-Led Projects", f"{df['WomenLed'].sum()} ({df['WomenLed'].sum()/len(df)*100:.1f}%)"],
                    ["Youth-Involved Projects", f"{df['YouthInvolved'].sum()} ({df['YouthInvolved'].sum()/len(df)*100:.1f}%)"],
                    ["Income to Women", f"${df['WomenIncomeShare'].sum():,.0f}"],
                ]
                
                impact_table = Table(impact_data, colWidths=[3.2*inch, 2.2*inch])
                impact_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.93, 0.26, 0.60)),
                    ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 12),
                    ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
                    ('FONTNAME', (1,1), (1,-1), 'Helvetica'),
                    ('FONTSIZE', (0,1), (-1,-1), 11),
                    ('TOPPADDING', (0,0), (-1,-1), 8),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
                    ('GRID', (0,0), (-1,-1), 1, rl_colors.grey),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ]))
                story.append(impact_table)
                story.append(Spacer(1, 24))

            # Regional Distribution
            story.append(Paragraph("<b>Regional Distribution</b>", heading_style))
            story.append(Spacer(1, 8))
            
            regional_df = df.groupby('Region').agg({
                'NetCredits': 'sum',
                'ProjectID': 'count',
                'MarketValue': 'sum'
            }).reset_index()
            
            regional_data = [["Region", "Projects", "Net Credits", "Market Value"]]
            for _, row in regional_df.iterrows():
                regional_data.append([
                    str(row['Region']),
                    f"{int(row['ProjectID']):,}",
                    f"{int(row['NetCredits']):,.0f}",
                    f"${int(row['MarketValue']):,.0f}"
                ])
            
            regional_table = Table(regional_data, colWidths=[1.8*inch, 1.2*inch, 1.5*inch, 1.5*inch])
            regional_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.23, 0.51, 0.96)),
                ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 11),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 10),
                ('TOPPADDING', (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('GRID', (0,0), (-1,-1), 1, rl_colors.grey),
                ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
                ('ALIGN', (0,0), (0,-1), 'LEFT'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.Color(0.97, 0.98, 0.97)])
            ]))
            story.append(regional_table)
            story.append(Spacer(1, 24))

            # Top 10 Projects Table
            story.append(Paragraph("<b>Top 10 Projects by Net Credits</b>", heading_style))
            story.append(Spacer(1, 10))
            
            if not df.empty:
                top = df.nlargest(10, 'NetCredits')[['ProjectID', 'Type', 'NetCredits', 'MarketValue', 'VerificationStatus']]
                table_data = [['Project ID', 'Type', 'Net Credits', 'Market Value', 'Status']]
                
                for _, row in top.iterrows():
                    table_data.append([
                        str(row['ProjectID'])[:20],
                        str(row['Type'])[:15],
                        f"{row['NetCredits']:,.0f}",
                        f"${row['MarketValue']:,.0f}",
                        str(row['VerificationStatus'])[:12]
                    ])
                
                top_table = Table(table_data, colWidths=[1.4*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.0*inch])
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), rl_colors.Color(0.06, 0.73, 0.51)),
                    ('TEXTCOLOR', (0,0), (-1,0), rl_colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 10),
                    ('BOTTOMPADDING', (0,0), (-1,0), 10),
                    ('TOPPADDING', (0,0), (-1,0), 10),
                    ('BACKGROUND', (0,1), (-1,-1), rl_colors.Color(0.97, 0.98, 0.97)),
                    ('GRID', (0,0), (-1,-1), 0.8, rl_colors.Color(0.85, 0.85, 0.85)),
                    ('FONTSIZE', (0,1), (-1,-1), 9),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_colors.white, rl_colors.Color(0.97, 0.98, 0.97)])
                ]))
                story.append(top_table)

            story.append(Spacer(1, 30))
            
            # Footer
            story.append(Paragraph(
                f"<i>© {datetime.now().year} GreenScope Analytics — Climate Intelligence Platform</i>",
                centered_style
            ))
            story.append(Paragraph(
                f"<i>Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}</i>",
                centered_style
            ))

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
        except Exception as e:
            logger.exception(f"PDF generation failed: {e}")
            st.error(f"PDF generation error: {str(e)}")
            return None


# ------------------------------
# Render Functions
# ------------------------------
def render_header():
    """Render modern header with branding"""
    st.markdown("""
        <div class="main-header">
            <div class="main-title">🌍 GreenScope Analytics</div>
            <div class="main-subtitle">Smart AI. Trusted Climate Impact.</div>
        </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame):
    """Render beautiful KPI metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Projects",
            f"{len(df):,}",
            delta=f"+{len(df[df['VerificationStatus']=='Verified'])} verified",
            delta_color="normal"
        )
    
    with col2:
        total_credits = df['NetCredits'].sum()
        st.metric(
            "Net Credits",
            f"{total_credits:,.0f}",
            delta="tCO₂e",
            delta_color="off"
        )
    
    with col3:
        avg_additionality = df['AdditionalityScore'].mean()
        st.metric(
            "Avg Additionality",
            f"{avg_additionality:.1f}%",
            delta=f"±{df['AdditionalityScore'].std():.1f}%",
            delta_color="off"
        )
    
    with col4:
        total_value = df['MarketValue'].sum()
        st.metric(
            "Market Value",
            f"${total_value/1e6:.1f}M",
            delta=f"${df['CO2Price'].mean():.0f}/tCO₂e avg",
            delta_color="off"
        )


def render_advanced_filters():
    """Render enhanced filter sidebar"""
    st.sidebar.markdown("## 🔍 Advanced Filters")
    
    with st.sidebar.expander("🌍 Geographic Filters", expanded=True):
        regions = st.multiselect(
            "Regions",
            sorted(st.session_state.df["Region"].unique()),
            default=sorted(st.session_state.df["Region"].unique()),
            key="region_filter"
        )
    
    with st.sidebar.expander("🏷️ Project Filters", expanded=True):
        types = st.multiselect(
            "Project Types",
            sorted(st.session_state.df["Type"].unique()),
            default=sorted(st.session_state.df["Type"].unique()),
            key="type_filter"
        )
        
        statuses = st.multiselect(
            "Verification Status",
            Config.VERIFICATION_STATUSES,
            default=Config.VERIFICATION_STATUSES,
            key="status_filter"
        )
        
        risks = st.multiselect(
            "Permanence Risk",
            ["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="risk_filter"
        )
    
    with st.sidebar.expander("📊 Metric Filters", expanded=True):
        credit_range = st.slider(
            "Net Credits (tCO₂e)",
            int(st.session_state.df["NetCredits"].min()),
            int(st.session_state.df["NetCredits"].max()),
            (int(st.session_state.df["NetCredits"].min()), 
             int(st.session_state.df["NetCredits"].max())),
            key="credit_range"
        )
        
        additionality_range = st.slider(
            "Additionality Score (%)",
            float(st.session_state.df["AdditionalityScore"].min()),
            100.0,
            (float(st.session_state.df["AdditionalityScore"].min()), 100.0),
            key="additionality_range"
        )
        
        vintage_range = st.slider(
            "Vintage Year",
            int(st.session_state.df["VintageYear"].min()),
            int(st.session_state.df["VintageYear"].max()),
            (int(st.session_state.df["VintageYear"].min()),
             int(st.session_state.df["VintageYear"].max())),
            key="vintage_range"
        )
    
    return regions, types, statuses, risks, credit_range, additionality_range, vintage_range


def render_dashboard_tab(df: pd.DataFrame):
    """Render enhanced dashboard with insights"""
    st.markdown("### 📊 Portfolio Overview")
    
    render_kpi_cards(df)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Regional Distribution")
        regional_df = df.groupby('Region').agg({
            'NetCredits': 'sum',
            'ProjectID': 'count',
            'MarketValue': 'sum'
        }).reset_index()
        regional_df.columns = ['Region', 'Total Credits', 'Project Count', 'Market Value']
        
        fig = px.bar(
            regional_df,
            x='Region',
            y='Total Credits',
            color='Market Value',
            title='',
            color_continuous_scale='Greens'
        )
        fig = VisualizationManager._apply_modern_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Risk Distribution")
        risk_counts = df['PermanenceRisk'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.6,
            marker=dict(colors=[Config.RISK_COLORS[risk] for risk in risk_counts.index]),
            textinfo='label+percent'
        )])
        fig = VisualizationManager._apply_modern_theme(fig)
        fig.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


def render_analytics_tab(df: pd.DataFrame):
    """Render comprehensive analytics"""
    st.markdown("### 📈 Advanced Analytics")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        fig1 = VisualizationManager.credits_by_type_enhanced(df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with viz_col2:
        fig2 = VisualizationManager.compliance_donut(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    fig3 = VisualizationManager.risk_bubble_chart(df)
    st.plotly_chart(fig3, use_container_width=True)


def render_map_tab(df: pd.DataFrame):
    """Render interactive map"""
    st.markdown("### 🗺️ Geographic Distribution")
    
    if df.empty:
        st.info("No projects to display on map")
        return
    
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        tiles="CartoDB positron"
    )
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in df.iterrows():
        color = "green" if row["VerificationStatus"] == "Verified" else "orange"
        popup_html = VisualizationManager.create_enhanced_popup(row)
        
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['ProjectID']} ({row['Type']})",
            icon=folium.Icon(color=color, icon="leaf", prefix="fa")
        ).add_to(marker_cluster)
    
    st_folium(m, width="100%", height=600)


def render_equity_ai_tab(df: pd.DataFrame):
    """Render AI-powered Equity Intelligence Dashboard"""
    st.markdown("### 🤖 AI Equity Intelligence")
    st.markdown("**Powered by machine learning to detect greenwashing and identify high-impact investments**")
    
    ai_system = EquityIntelligenceAI()
    
    with st.spinner("🧠 AI analyzing equity metrics across portfolio..."):
        df_equity = df.copy()
        df_equity = ai_system.detect_social_impact_greenwashing(df_equity)
        df_equity = ai_system.calculate_equity_excellence_score(df_equity)
        df_equity = ai_system.predict_impact_potential(df_equity)
    
    st.markdown("---")
    st.markdown("### 🚨 AI Alert Summary")
    
    alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
    
    with alert_col1:
        flagged_count = df_equity['IsSociallyFlagged'].sum()
        flagged_pct = (flagged_count / len(df_equity)) * 100
        st.metric(
            "⚠️ Flagged Projects",
            f"{flagged_count}",
            f"{flagged_pct:.1f}% of portfolio",
            delta_color="inverse"
        )
    
    with alert_col2:
        total_flags = df_equity['RedFlags'].apply(len).sum()
        st.metric(
            "🚩 Total Red Flags",
            f"{total_flags}",
            "Issues detected"
        )
    
    with alert_col3:
        champions_count = len(df_equity[df_equity['EquityExcellenceScore'] > 80])
        st.metric(
            "⭐ Equity Champions",
            f"{champions_count}",
            "Score > 80"
        )
    
    with alert_col4:
        avg_score = df_equity['EquityExcellenceScore'].mean()
        st.metric(
            "📊 Avg Equity Score",
            f"{avg_score:.1f}",
            "Portfolio average"
        )
    
    if flagged_count > 0:
        st.markdown("---")
        st.markdown("### ⚠️ Projects Requiring Review")
        st.warning(f"AI has flagged {flagged_count} projects with potential social impact inconsistencies")
        
        flagged_df = df_equity[df_equity['IsSociallyFlagged'] == True].copy()
        flagged_df['FlagCount'] = flagged_df['RedFlags'].apply(len)
        
        display_flagged = flagged_df[['ProjectID', 'Type', 'Region', 'NetCredits', 
                                       'EquityExcellenceScore', 'FlagCount', 'RedFlags']].sort_values('FlagCount', ascending=False)
        
        for idx, row in display_flagged.head(5).iterrows():
            with st.expander(f"🚩 {row['ProjectID']} - {row['Type']} ({row['FlagCount']} flags)"):
                st.write(f"**Region:** {row['Region']}")
                st.write(f"**Net Credits:** {row['NetCredits']:,.0f} tCO₂e")
                st.write(f"**Equity Excellence Score:** {row['EquityExcellenceScore']:.1f}/100")
                st.write("**AI-Detected Issues:**")
                for flag in row['RedFlags']:
                    st.error(flag)
                
                st.info("💡 **Recommendation:** Request additional documentation on social impact claims before investment")
    
    st.markdown("---")
    st.markdown("### 📊 Portfolio Equity Excellence Distribution")
    
    fig1 = go.Figure()
    bins = [0, 40, 60, 80, 100]
    labels = ['Poor (0-40)', 'Fair (40-60)', 'Good (60-80)', 'Excellent (80-100)']
    colors = ['#ef4444', '#f59e0b', '#10b981', '#8b5cf6']
    
    for i in range(len(bins)-1):
        bin_data = df_equity[(df_equity['EquityExcellenceScore'] >= bins[i]) & 
                             (df_equity['EquityExcellenceScore'] < bins[i+1])]
        fig1.add_trace(go.Bar(
            name=labels[i],
            x=[labels[i]],
            y=[len(bin_data)],
            marker_color=colors[i],
            text=len(bin_data),
            textposition='outside'
        ))
    
    fig1.update_layout(
        title="AI-Generated Equity Excellence Scores",
        xaxis_title="Score Range",
        yaxis_title="Number of Projects",
        showlegend=False,
        height=400
    )
    fig1 = VisualizationManager._apply_modern_theme(fig1)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### ⭐ AI-Identified Equity Champions")
    st.success("These projects demonstrate the highest combination of carbon impact and social equity")
    
    champions = ai_system.identify_equity_champions(df_equity, top_n=5)
    
    for idx, row in champions.iterrows():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                        color: white; padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: white;">🏆 {row['ProjectID']} - {row['Type']}</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">
                    {row['Location']} | Equity Excellence Score: {row['EquityExcellenceScore']:.1f}/100
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
            with impact_col1:
                st.metric("Carbon Credits", f"{row['NetCredits']:,.0f}")
            with impact_col2:
                st.metric("Women Beneficiaries", f"{row['WomenBeneficiaries']:,}")
            with impact_col3:
                st.metric("Youth Employed", f"{row['YouthJobs']:,}")
            with impact_col4:
                st.metric("Local Income", f"${row['WomenIncomeShare']:,.0f}")
        
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row['EquityExcellenceScore'],
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': "#fee2e2"},
                        {'range': [40, 60], 'color': "#fef3c7"},
                        {'range': [60, 80], 'color': "#d1fae5"},
                        {'range': [80, 100], 'color': "#a7f3d0"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🎯 AI Impact Potential Analysis")
    st.info("AI predicts which projects could maximize impact with targeted interventions")
    
    potential_counts = df_equity['ImpactPotential'].value_counts()
    
    potential_col1, potential_col2 = st.columns(2)
    
    with potential_col1:
        fig2 = go.Figure(data=[go.Pie(
            labels=potential_counts.index,
            values=potential_counts.values,
            hole=0.4,
            marker=dict(colors=['#10b981', '#f59e0b', '#8b5cf6', '#3b82f6']),
            textinfo='label+percent'
        )])
        fig2.update_layout(title="Project Potential Distribution", height=400)
        fig2 = VisualizationManager._apply_modern_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    
    with potential_col2:
        st.markdown("#### 💡 AI Recommendations")
        
        high_potential = len(df_equity[df_equity['ImpactPotential'] == 'High Potential - Needs Social Focus'])
        equity_leaders = len(df_equity[df_equity['ImpactPotential'] == 'Equity Leader - Scale Carbon'])
        balanced = len(df_equity[df_equity['ImpactPotential'] == '⭐ Balanced Excellence'])
        
        st.markdown(f"""
        **{high_potential} projects** have strong carbon performance but could improve social impact.
        
        **{equity_leaders} projects** are equity leaders but could scale carbon impact.
        
        **{balanced} projects** demonstrate excellence in both dimensions.
        """)
    
    st.markdown("---")
    st.markdown("### 📥 Export AI Analysis")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        ai_report = df_equity[['ProjectID', 'Type', 'Region', 'NetCredits', 
                                'EquityExcellenceScore', 'IsSociallyFlagged', 
                                'ImpactPotential']].copy()
        csv = ai_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            "🤖 Download AI Analysis (CSV)",
            data=csv,
            file_name=f"equity_ai_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col2:
        exec_summary = f"""
GREENSCOPE EQUITY INTELLIGENCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

PORTFOLIO OVERVIEW
Total Projects Analyzed: {len(df_equity)}
Average Equity Excellence Score: {df_equity['EquityExcellenceScore'].mean():.1f}/100

RISK ALERTS
Projects Flagged by AI: {df_equity['IsSociallyFlagged'].sum()}
Total Red Flags Detected: {df_equity['RedFlags'].apply(len).sum()}

SOCIAL IMPACT TOTALS
Women Beneficiaries: {df_equity['WomenBeneficiaries'].sum():,}
Youth Beneficiaries: {df_equity['YouthBeneficiaries'].sum():,}
Jobs for Women: {df_equity['WomenJobs'].sum():,}
Jobs for Youth: {df_equity['YouthJobs'].sum():,}

Powered by GreenScope Equity Intelligence AI
        """
        
        st.download_button(
            "📋 Download Executive Summary (TXT)",
            data=exec_summary,
            file_name=f"equity_executive_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def render_compare_tab(df: pd.DataFrame):
    """Render project comparison"""
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
            st.metric("Net Credits", f"{d2['NetCredits']:,.0f} tCO₂e", 
                     delta=f"{delta_credits:+,.0f}")
            
            delta_value = d2['MarketValue'] - d1['MarketValue']
            st.metric("Market Value", f"${d2['MarketValue']:,.0f}",
                     delta=f"${delta_value:+,.0f}")
            
            delta_add = d2['AdditionalityScore'] - d1['AdditionalityScore']
            st.metric("Additionality Score", f"{d2['AdditionalityScore']:.1f}%",
                     delta=f"{delta_add:+.1f}%")


def render_report_tab(df: pd.DataFrame):
    """Render report generation interface"""
    st.markdown("### 📋 Generate Portfolio Report")
    
    st.info("📄 Generate a comprehensive PDF report with all visualizations, analytics, and project summaries.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Report Statistics")
        st.write(f"- **Projects in Report:** {len(df)}")
        st.write(f"- **Total Credits:** {df['NetCredits'].sum():,.0f} tCO₂e")
        st.write(f"- **Total Value:** ${df['MarketValue'].sum():,.0f}")
        
        if 'WomenBeneficiaries' in df.columns:
            st.write(f"- **Women Beneficiaries:** {df['WomenBeneficiaries'].sum():,}")
            st.write(f"- **Youth Employed:** {df['YouthJobs'].sum():,}")
    
    with col2:
        st.markdown("#### Quick Export")
        
        # CSV Export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV Data",
            data=csv,
            file_name=f"greenscope_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Excel Export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Projects', index=False)
        excel_buffer.seek(0)
        
        st.download_button(
            "📊 Download Excel Report",
            data=excel_buffer,
            file_name=f"greenscope_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # PDF Generation
    st.markdown("#### 📄 Generate Branded PDF Report")
    
    if st.button("🎨 Generate Professional PDF", type="primary", use_container_width=True):
        with st.spinner("Generating comprehensive report..."):
            try:
                pdf_bytes = ReportGenerator.generate_pdf(df)
                
                if pdf_bytes:
                    st.success("✅ Report generated successfully!")
                    st.download_button(
                        "📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"GreenScope_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("❌ Failed to generate report. Please check the logs.")
            except Exception as e:
                st.error(f"❌ Error generating PDF: {str(e)}")
                st.info("💡 Try downloading CSV or Excel format instead.")


def main():
    """Main application entry point"""
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
        with st.spinner("📄 Loading portfolio data..."):
            seed = 42 if fixed_dataset else None
            st.session_state.df = CarbonProjectDataManager.generate_realistic_carbon_projects(
                n_projects=n_projects,
                seed=seed
            )
            st.session_state.df = generate_impact_data(st.session_state.df)
    
    df = st.session_state.df
    
    validation = CarbonProjectDataManager.validate_dataframe(df)
    
    if not validation["valid"]:
        st.error("⚠️ Data Validation Issues Detected:")
        for issue in validation["issues"]:
            st.error(f"- {issue}")
        return
    
    regions, types, statuses, risks, credit_range, additionality_range, vintage_range = render_advanced_filters()
    
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if 'filter' in key or 'range' in key:
                del st.session_state[key]
        st.rerun()
    
    filtered_df = apply_filters(
        df, regions, types, statuses, risks,
        credit_range, additionality_range, vintage_range
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📊 Filtered Results")
    st.sidebar.metric("Projects Shown", f"{len(filtered_df):,}")
    st.sidebar.metric("Total Credits", f"{filtered_df['NetCredits'].sum():,.0f}")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No projects match your current filters. Please adjust your criteria.")
        return
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard",
        "📈 Analytics",
        "🗺️ Geographic Map",
        "🤖 AI Equity Intelligence",
        "⚖️ Compare Projects",
        "📋 Reports",
        "📑 Data Table"
    ])
    
    with tab1:
        render_dashboard_tab(filtered_df)
    
    with tab2:
        render_analytics_tab(filtered_df)
    
    with tab3:
        render_map_tab(filtered_df)
    
    with tab4:
        render_equity_ai_tab(filtered_df)
    
    with tab5:
        render_compare_tab(filtered_df)
    
    with tab6:
        render_report_tab(filtered_df)
    
    with tab7:
        st.markdown("### 📑 Project Data Table")
        
        all_columns = filtered_df.columns.tolist()
        default_columns = ['ProjectID', 'Type', 'Region', 'NetCredits', 'VerificationStatus', 
                          'PermanenceRisk', 'AdditionalityScore', 'MarketValue']
        
        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if selected_columns:
            search_term = st.text_input("🔍 Search Projects", "")
            
            display_df = filtered_df[selected_columns].copy()
            
            if search_term:
                mask = display_df.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False, na=False).any(),
                    axis=1
                )
                display_df = display_df[mask]
            
            col1, col2 = st.columns([3, 1])
            with col1:
                sort_by = st.selectbox("Sort By", selected_columns)
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            
            display_df = display_df.sort_values(
                by=sort_by,
                ascending=(sort_order == "Ascending")
            )
            
            format_dict = {}
            format_specs = {
                'NetCredits': '{:,.0f}',
                'TotalCredits': '{:,.0f}',
                'BufferCredits': '{:,.0f}',
                'BaselineEmissions': '{:,.0f}',
                'AdditionalityScore': '{:.1f}%',
                'MarketValue': '${:,.0f}',
                'CO2Price': '${:.2f}',
                'WomenBeneficiaries': '{:,}',
                'YouthBeneficiaries': '{:,}',
                'WomenJobs': '{:,}',
                'YouthJobs': '{:,}',
                'WomenIncomeShare': '${:,.0f}',
                'EquityExcellenceScore': '{:.1f}'
            }
            
            for col_name, fmt in format_specs.items():
                if col_name in display_df.columns:
                    format_dict[col_name] = fmt
            
            st.dataframe(
                display_df.style.format(format_dict),
                use_container_width=True,
                height=500
            )
            
            st.caption(f"Showing {len(display_df)} of {len(filtered_df)} projects")
        else:
            st.info("Please select at least one column to display")
    
    st.markdown("---")
    st.markdown(f"""
        <div class="footer">
            <p><strong>GreenScope Analytics</strong> — Climate Intelligence Platform</p>
            <p>Showing {len(filtered_df):,} of {len(df):,} projects | 
               Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')} | 
               Total Portfolio Value: ${filtered_df['MarketValue'].sum():,.0f}</p>
            <p style="font-size: 0.75rem; margin-top: 1rem;">
                © {datetime.now().year} GreenScope Analytics. All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()