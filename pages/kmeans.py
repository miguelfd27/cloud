import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import streamlit as st

st.title("Kmeans")

data = pd.read_csv("C:/Users/mfdourado.INDRA/Documents/jupyter/Placement_Data_Full_Class.csv")
data["salary"] = data["salary"].fillna(0)

categorical = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
le = LabelEncoder()
for col in categorical:
    data[col] = le.fit_transform(data[col])

data = data.drop(["sl_no", "gender", "status"], axis=1)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

data_scaled_df = pd.DataFrame(data_scaled, columns=data.columns)

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled_df)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('SSE')
st.pyplot(plt)

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled_df)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled_df)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = data['Cluster']

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('Clusters de Estudiantes')
st.pyplot(plt)

cluster_analysis = data.groupby('Cluster').mean()
st.write(cluster_analysis)
