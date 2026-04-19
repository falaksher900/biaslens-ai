import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BiasLens AI", layout="wide")

st.title("📊 BiasLens AI - Behavioural Analytics Platform")

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload your Excel dataset", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ===============================
    # MODE SELECTION
    # ===============================
    mode = st.selectbox("Select Analysis Mode", 
                        ["Behavioural Analysis", "Risk Profiling", "Performance Clustering"])

    # ===============================
    # FEATURE SELECTION
    # ===============================
    st.subheader("📊 Select Features for Clustering")
    selected_features = st.multiselect("Choose columns", df.columns)

    if len(selected_features) > 1:

        features = df[selected_features]

        # ===============================
        # SCALING
        # ===============================
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)

        # ===============================
        # CLUSTER INPUT
        # ===============================
        k = st.slider("Select number of clusters", 2, 6, 3)

        if st.button("Run Analysis"):

            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)

            st.success("Clustering Completed!")

            # ===============================
            # VISUALS
            # ===============================
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📍 Cluster Scatter Plot")
                fig, ax = plt.subplots()
                ax.scatter(scaled_data[:,0], scaled_data[:,1], c=df['Cluster'])
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                st.pyplot(fig)

            with col2:
                st.subheader("📊 Cluster Distribution")
                st.bar_chart(df['Cluster'].value_counts())

            # Heatmap
            st.subheader("🔥 Cluster Behaviour Heatmap")
            heatmap_data = df.groupby('Cluster')[selected_features].mean()
            fig2, ax2 = plt.subplots()
            sns.heatmap(heatmap_data, annot=True, ax=ax2)
            st.pyplot(fig2)

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
                    return "Overconfidence Bias"
                elif score <= -2:
                    return "Loss Aversion Bias"
                elif score < 0:
                    return "Risk-Averse Behaviour"
                else:
                    return "Balanced Behaviour"

            df['Behaviour_Type'] = df.apply(classify, axis=1)

            st.subheader("🧠 Behaviour Classification")
            st.dataframe(df[['Cluster','Behaviour_Type']])

            # ===============================
            # INSIGHTS
            # ===============================
            st.subheader("📌 Cluster Insights")

            for cluster in sorted(df['Cluster'].unique()):
                cluster_data = df[df['Cluster'] == cluster]

                if 'Risk_Level' in df.columns and 'Trade_Frequency' in df.columns:
                    if cluster_data['Risk_Level'].mean() > df['Risk_Level'].mean():
                        insight = "High-risk investors → Possible OVERCONFIDENCE"
                    else:
                        insight = "Moderate or low-risk behaviour → BALANCED"

                st.write(f"Cluster {cluster}: {insight}")
