import streamlit as st
import pandas as pd
import joblib
from modelo import cargar_modelo

st.set_page_config(page_title="Clasificador de Enfermedades de Soya", layout="wide")

st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTjIZOG3YgrUv-eWxVSq7qYNknj-NV6tzNFAw&s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #2e7d32;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }

    .stButton>button {
        background-color: #2e7d32;
        color: black;
        border: none;
        padding: 0.6rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title("🌿 Clasificador de Enfermedades en Plantas de Soya")

try:
    model, scaler, encoder = cargar_modelo()
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

st.subheader("🧪 Ingresa los síntomas observados:")

with st.form("diagnostic_form"):
    cols = st.columns(3)
    campos = [
        "date", "plant-stand", "precip", "temp", "hail", "crop-hist", "area-damaged", "severity", 
        "seed-tmt", "germination", "plant-growth", "leaves", "leafspots-halo", "leafspots-marg", 
        "leafspot-size", "leaf-shread", "leaf-malf", "leaf-mild", "stem", "lodging", "stem-cankers", 
        "canker-lesion", "fruiting-bodies", "external-decay", "mycelium", "int-discolor", "sclerotia", 
        "fruit-pods", "fruit-spots", "seed", "mold-growth", "seed-discolor", "seed-size", "shriveling", "roots"
    ]
    
    opciones = {
        "date": ["abril", "mayo", "junio", "julio"],
        "plant-stand": ["normal", "lt-normal"],
        "precip": ["lt-norm", "norm", "gt-norm"],
        "temp": ["lt-norm", "norm", "gt-norm"],
        "hail": ["yes", "no"],
        "crop-hist": ["continuous", "previous", "none"],
        "area-damaged": ["none", "low", "medium", "high"],
        "severity": ["none", "low", "medium", "high"],
        "seed-tmt": ["none", "tmt1", "tmt2", "tmt3"],
        "germination": ["low", "high"],
        "plant-growth": ["normal", "stunted", "luxuriant"],
        "leaves": ["normal", "abnormal"],
        "leafspots-halo": ["yes", "no"],
        "leafspots-marg": ["yes", "no"],
        "leafspot-size": ["small", "medium", "large"],
        "leaf-shread": ["yes", "no"],
        "leaf-malf": ["yes", "no"],
        "leaf-mild": ["yes", "no"],
        "stem": ["normal", "abnormal"],
        "lodging": ["yes", "no"],
        "stem-cankers": ["yes", "no"],
        "canker-lesion": ["yes", "no"],
        "fruiting-bodies": ["yes", "no"],
        "external-decay": ["yes", "no"],
        "mycelium": ["yes", "no"],
        "int-discolor": ["yes", "no"],
        "sclerotia": ["yes", "no"],
        "fruit-pods": ["none", "few", "some", "many"],
        "fruit-spots": ["yes", "no"],
        "seed": ["normal", "abnormal"],
        "mold-growth": ["yes", "no"],
        "seed-discolor": ["yes", "no"],
        "seed-size": ["small", "medium", "large"],
        "shriveling": ["yes", "no"],
        "roots": ["normal", "abnormal"]
    }

    respuestas = {}
    for i, campo in enumerate(campos):
        with cols[i % 3]:
            respuestas[campo] = st.selectbox(campo.replace("-", " ").capitalize(), opciones[campo])

    submitted = st.form_submit_button("Diagnosticar")

if submitted:
    try:
        input_data = pd.DataFrame({k: [v] for k, v in respuestas.items()})
        X_encoded = encoder.transform(input_data)
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(input_data.columns))
        X_scaled = scaler.transform(X_encoded_df)

        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled).max()

        st.markdown(f"""
        <div style='text-align: center; margin-top: 30px;'>
            <h2 style='color: #1b5e20;'>Diagnóstico: <b>{prediction[0]}</b></h2>
            <p style='font-size: 1.2rem;'>Confianza del modelo: <b>{probability:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)
