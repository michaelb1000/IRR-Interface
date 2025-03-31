import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from PIL import Image
from streamlit.components.v1 import html

# Load model and data
MODEL_PATH = "model.pkl2"
csv_path = "ML_Data_CSV_2.csv"
model = joblib.load(MODEL_PATH)
df = pd.read_csv(csv_path)

# Extract UI options
opportunity_types = list(df["OpportunityType"].dropna().unique())
states = list(df["State"].dropna().unique())
offtakerectors = list(df["OfftakerSector"].dropna().unique())
funds = list(df["Fund"].dropna().unique())

# Encoding maps
opportunity_type_map = {val: idx for idx, val in enumerate(opportunity_types)}
state_map = {val: idx for idx, val in enumerate(states)}
offtakerector_map = {val: idx for idx, val in enumerate(offtakerectors)}
fund_map = {val: idx for idx, val in enumerate(funds)}

def encode_input(opportunity_type, state, offtaker_sector, fund, epc, size):
    return [
        opportunity_type_map.get(opportunity_type, -1),
        state_map.get(state, -1),
        offtakerector_map.get(offtaker_sector, -1),
        fund_map.get(fund, -1),
        epc,
        size
    ]

# Load logo image
logo = Image.open("onyx_logo_vertical_black.png")

# Custom CSS
st.markdown("""
    <style>
        .gradient-bar {
            height: 10px;
            background: linear-gradient(to right, #f8e71c, #f5a623, #f56f1c, #d0021b);
            border-radius: 5px;
            margin-top: 10px;
            margin-bottom: 50px;
        }
        .header-text {
            font-size: 36px;
            font-weight: 700;
            color: #202020;
            padding-top: 10px;
        }
        .side-gradient {
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            width: 12px;
            background: linear-gradient(to bottom, #a5dc32, #f8e71c, #f5a623, #f56f1c, #d0021b);
            z-index: 999;
        }
        .section-space {
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
            margin-bottom: -10px;
        }
    </style>
    <div class="side-gradient"></div>
""", unsafe_allow_html=True)

# Header layout
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="header-text">Investment IRR Prediction</div>', unsafe_allow_html=True)
with col2:
    st.image(logo, width=100)

st.markdown('<div class="gradient-bar"></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Enter Investment Details")
opportunity_type = st.sidebar.selectbox("Opportunity Type", opportunity_types + ["Other"])
state = st.sidebar.selectbox("State", states + ["Other"])
offtaker_sector = st.sidebar.selectbox("Offtaker Sector", offtakerectors + ["Other"])
fund = st.sidebar.selectbox("Fund", funds + ["Other"])
size = st.sidebar.number_input("System Size (kw)", min_value=0.0, max_value=100000.0, value=500.0)
epc = st.sidebar.number_input("EPC", min_value=0.0, max_value=10.0, value=2.5)

# Plot checkboxes (including IRR Map)
st.sidebar.markdown("### Select Plots to Show:")
show_prediction = st.sidebar.checkbox("Show Predicted IRR", value=True)
show_feature_importance = st.sidebar.checkbox("Show Feature Importance")
show_scatter = st.sidebar.checkbox("Show Scatter Plot")
show_pdp = st.sidebar.checkbox("Show Partial Dependence Plot")
show_map = st.sidebar.checkbox("Show IRR Map", value=False)

encoded_input = np.array([encode_input(opportunity_type, state, offtaker_sector, fund, epc, size)], dtype=np.float32)

def streamlit_scatter_plot():
    test_df = df.dropna(subset=['EPC', 'Size']).copy()
    test_df['OpportunityType'] = test_df['OpportunityType'].map(opportunity_type_map).fillna(-1)
    test_df['State'] = test_df['State'].map(state_map).fillna(-1)
    test_df['OfftakerSector'] = test_df['OfftakerSector'].map(offtakerector_map).fillna(-1)
    test_df['Fund'] = test_df['Fund'].map(fund_map).fillna(-1)
    X_test = test_df[["OpportunityType", "State", "OfftakerSector", "Fund", "EPC", "Size"]].astype(float)
    predicted_irr = model.predict(X_test) * 100
    test_df['Predicted IRR'] = predicted_irr
    user_predicted_irr = model.predict(encoded_input)[0] * 100

    st.markdown("<div class='section-space'>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(test_df['EPC'], test_df['Size'], c=test_df['Predicted IRR'], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Predicted IRR (%)')
    ax.scatter(encoded_input[0][4], encoded_input[0][5], c='red', s=100, edgecolors='black', label=f'User Predicted: {user_predicted_irr:.2f}%')
    ax.set_xlabel('EPC')
    ax.set_ylabel('Size')
    ax.set_title('Predicted IRR — Scatter Plot')
    ax.legend()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

if st.sidebar.button("Predict IRR"):
    predicted_irr_value = model.predict(encoded_input)[0] * 100

    if show_prediction:
        st.markdown("<div class='section-title'>Random Forest Prediction Output</div>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="text-align: center; font-size: 24px; font-weight: bold; margin-top: 10px;">
                Predicted IRR: <span style='border: 2px dashed red; padding: 5px;'>{predicted_irr_value:.2f}%</span>
            </div>
        """, unsafe_allow_html=True)

    if show_scatter:
        streamlit_scatter_plot()

    if show_feature_importance or show_pdp:
        st.markdown("<div class='section-title'>Parameter Analysis</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        if show_feature_importance:
            with col1:
                fig, ax = plt.subplots()
                feature_names = ["Opportunity Type", "State", "Offtaker Sector", "Fund", "EPC", "Size"]
                ax.barh(feature_names, model.feature_importances_, color='skyblue')
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

        if show_pdp:
            with col2:
                with st.spinner("Generating Partial Dependence Plot..."):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    prepared_df = df.copy()
                    prepared_df['OpportunityType'] = prepared_df['OpportunityType'].map(opportunity_type_map).fillna(-1)
                    prepared_df['State'] = prepared_df['State'].map(state_map).fillna(-1)
                    prepared_df['OfftakerSector'] = prepared_df['OfftakerSector'].map(offtakerector_map).fillna(-1)
                    prepared_df['Fund'] = prepared_df['Fund'].map(fund_map).fillna(-1)
                    X_train = prepared_df[["OpportunityType", "State", "OfftakerSector", "Fund", "EPC", "Size"]].astype(float)
                    display = PartialDependenceDisplay.from_estimator(model, X_train, features=["EPC", "Size"], ax=ax)
                    ax.set_title("Partial Dependence Plot")
                    st.pyplot(fig)

# Show IRR Map (after Predict IRR section)
if show_map:
    st.markdown("<div class='section-title'>State-Level Average IRR Map</div>", unsafe_allow_html=True)

    import plotly.graph_objects as go

    state_irr = df.groupby("State")["IRR"].mean().reset_index()
    state_irr_dict = dict(zip(state_irr["State"], state_irr["IRR"]))

    all_state_abbr = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]

    hover_text = []
    z_values = []

    for state in all_state_abbr:
        if state in state_irr_dict:
            irr_value = round(state_irr_dict[state] * 100, 1)
            hover_text.append(f"{irr_value}%")
            z_values.append(1)
        else:
            hover_text.append("")
            z_values.append(0)

    fig = go.Figure(data=go.Choropleth(
        locations=all_state_abbr,
        z=z_values,
        locationmode='USA-states',
        colorscale=[[0, 'lightgray'], [1, 'blue']],
        showscale=False,
        text=hover_text,
        hoverinfo='text'
    ))

    fig.update_layout(
        geo=dict(scope='usa'),
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig, use_container_width=True)


# Run with: streamlit run RFR_Interface.py