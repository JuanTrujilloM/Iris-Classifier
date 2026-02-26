import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------
st.set_page_config(page_title="Clasificador IRIS", layout="wide")
st.title("🌸 Clasificador Dinámico - Dataset IRIS")
st.markdown("Aplicación pedagógica para entrenamiento, evaluación y predicción manual.")

# ------------------------------
# CARGA DE DATOS
# ------------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = y

# ------------------------------
# SIDEBAR - CONFIGURACIÓN
# ------------------------------
st.sidebar.header("⚙ Configuración del Modelo")

test_size = st.sidebar.slider("Proporción de prueba (%)", 10, 50, 30) / 100

model_option = st.sidebar.selectbox(
    "Seleccione el modelo",
    ["KNN", "Decision Tree", "Logistic Regression"]
)

# Parámetros dinámicos
if model_option == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 15, 3)

if model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)

# ------------------------------
# VISUALIZACIÓN DEL DATASET
# ------------------------------
st.subheader("📊 Exploración del Dataset")

col1, col2 = st.columns(2)

with col1:
    st.write("Primeras filas del dataset")
    st.dataframe(df.head())

with col2:
    fig = px.scatter(
        df,
        x=feature_names[0],
        y=feature_names[1],
        color=df["species"].map(dict(enumerate(target_names))),
        title="Distribución de Clases"
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# DIVISIÓN DE DATOS
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# ENTRENAMIENTO
# ------------------------------
if model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=max_depth)

else:
    model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ------------------------------
# RESULTADOS
# ------------------------------
st.subheader("📈 Desempeño del Modelo")

col3, col4 = st.columns(2)

with col3:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col4:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=target_names,
                yticklabels=target_names,
                cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig_cm)

st.subheader("📋 Reporte de Clasificación")
st.text(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------------
# INTERFAZ DE PREDICCIÓN MANUAL
# ------------------------------
st.subheader("🔎 Predicción Manual")

st.markdown("Ingrese valores para predecir la especie:")

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()))
    input_data.append(value)

if st.button("Predecir"):
    input_array = np.array(input_data).reshape(1, -1)
    input_array = scaler.transform(input_array)
    prediction = model.predict(input_array)
    predicted_class = target_names[prediction[0]]

    st.success(f"La especie predicha es: **{predicted_class.upper()}** 🌸")

# ------------------------------
# SECCIÓN PEDAGÓGICA
# ------------------------------
st.subheader("📘 Explicación Pedagógica")

if model_option == "KNN":
    st.info("""
    KNN clasifica un nuevo punto según la mayoría de sus vecinos más cercanos.
    Es un modelo basado en distancia.
    """)

elif model_option == "Decision Tree":
    st.info("""
    Decision Tree crea reglas de decisión basadas en divisiones sucesivas.
    Es interpretable y basado en reglas.
    """)

else:
    st.info("""
    Logistic Regression usa una función logística para modelar probabilidades.
    Es un modelo lineal probabilístico.
    """)
