import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="BiasLens AI", layout="wide")

# ===============================
# HEADER
# ===============================
st.markdown("""
# 📊 BiasLens AI
### Behavioural Analytics Dashboard
""")

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx"])

mode = st.sidebar.selectbox(
    "Analysis Mode",
    ["Behavioural Analysis", "Risk Profiling", "Performance Clustering"]
)

k = st.sidebar.slider("Number of Clusters", 2, 6, 3)

# ===============================
# MAIN LOGIC
# ===============================
if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ===============================
    # FEATURE SELECTION
    # ===============================
    selected_features = st.multiselect(
        "Select Features for Analysis",
        df.columns,
        default=df.columns[:3]
    )

    if len(selected_features) > 1:

        features = df[selected_features]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)

        if st.button("🚀 Run Analysis"):

            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)

            # ===============================
            # KPI CARDS
            # ===============================
            col1, col2, col3 = st.columns(3)

            col1.metric("Total Investors", len(df))
            col2.metric("Clusters", k)
            col3.metric("Features Used", len(selected_features))

            st.markdown("---")

            # ===============================
            # VISUALS (PROPER GRID)
            # ===============================
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📍 Cluster Map")
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(scaled_data[:,0], scaled_data[:,1], c=df['Cluster'])
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Cluster Distribution")
                st.bar_chart(df['Cluster'].value_counts())

            # ===============================
            # HEATMAP (FIXED SIZE)
            # ===============================
            st.subheader("🔥 Behaviour Heatmap")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            sns.heatmap(df.groupby('Cluster')[selected_features].mean(), annot=True, ax=ax2)
            st.pyplot(fig2)

            st.markdown("---")

            # ===============================
            # BEHAVIOUR CLASSIFICATION
            # ===============================
            def classify(row):
                score = 0

                if 'Trade_Frequency' in df.columns and row.get('Trade_Frequency', 0) > df['Trade_Frequency'].mean():
                    score += 2
                if 'Risk_Level' in df.columns and row.get('Risk_Level', 0) > df['Risk_Level'].mean():
                    score += 2
                if 'Profit_Loss' in df.columns and row.get('Profit_Loss', 0) > df['Profit_Loss'].mean():
                    score += 1
                if 'Holding_Period' in df.columns and row.get('Holding_Period', 0) > df['Holding_Period'].mean():
                    score -= 2
                if 'Drawdown' in df.columns and row.get('Drawdown', 0) > df['Drawdown'].mean():
                    score -= 2

                if score >= 3:
                    return "Overconfidence"
                elif score <= -2:
                    return "Loss Aversion"
                elif score < 0:
                    return "Risk-Averse"
                else:
                    return "Balanced"

            df['Behaviour'] = df.apply(classify, axis=1)

            st.subheader("🧠 Behaviour Insights")
            st.dataframe(df[['Cluster','Behaviour']], use_container_width=True)

            # ===============================
            # DOWNLOAD BUTTON ✅
            # ===============================
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇️ Download Results",
                csv,
                "biaslens_results.csv",
                "text/csv"
            )
