import streamlit as st
import pandas as pd
import joblib
from modelo import cargar_modelo

# Configuración de la página
st.set_page_config(page_title="Clasificador de Enfermedades de Soja", layout="wide")

# Título de la aplicación
st.title("Sistema de Diagnóstico de Enfermedades en Plantas de Soja")

# Cargar modelo
try:
    model, scaler, encoder = cargar_modelo()
    st.success("Modelo cargado correctamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {str(e)}")
    st.stop()

# Sección para ingresar datos
st.header("Ingrese los síntomas observados")

# Crear un formulario para ingresar los datos
with st.form("diagnostic_form"):
    date = st.selectbox("Fecha de observación", ["abril", "mayo", "junio", "julio"])
    plant_stand = st.selectbox("Porte de la planta", ["normal", "lt-normal"])
    precip = st.selectbox("Precipitación", ["lt-norm", "norm", "gt-norm"])
    temp = st.selectbox("Temperatura", ["lt-norm", "norm", "gt-norm"])
    hail = st.selectbox("Granizo", ["yes", "no"])
    crop_hist = st.selectbox("Historial de cultivo", ["continuous", "previous", "none"])
    area_damaged = st.selectbox("Área dañada", ["none", "low", "medium", "high"])
    severity = st.selectbox("Severidad", ["none", "low", "medium", "high"])
    seed_tmt = st.selectbox("Tratamiento de semillas", ["none", "tmt1", "tmt2", "tmt3"])
    germination = st.selectbox("Germinación", ["low", "high"])
    plant_growth = st.selectbox("Crecimiento de la planta", ["normal", "stunted", "luxuriant"])
    leaves = st.selectbox("Hojas", ["normal", "abnormal"])
    leafspots_halo = st.selectbox("Manchas en hojas - halo", ["yes", "no"])
    leafspots_marg = st.selectbox("Manchas en hojas - margen", ["yes", "no"])
    leafspot_size = st.selectbox("Tamaño de las manchas en hojas", ["small", "medium", "large"])
    leaf_shread = st.selectbox("Desgarro de hojas", ["yes", "no"])
    leaf_malf = st.selectbox("Malformación de hojas", ["yes", "no"])
    leaf_mild = st.selectbox("Manchas leves en hojas", ["yes", "no"])
    stem = st.selectbox("Tallos", ["normal", "abnormal"])
    lodging = st.selectbox("Alojamientos", ["yes", "no"])
    stem_cankers = st.selectbox("Cánceres en el tallo", ["yes", "no"])
    canker_lesion = st.selectbox("Lesión en cáncer", ["yes", "no"])
    fruiting_bodies = st.selectbox("Cuerpos fructíferos", ["yes", "no"])
    external_decay = st.selectbox("Decaimiento externo", ["yes", "no"])
    mycelium = st.selectbox("Micelio", ["yes", "no"])
    int_discolor = st.selectbox("Descoloración interna", ["yes", "no"])
    sclerotia = st.selectbox("Esclerotia", ["yes", "no"])
    fruit_pods = st.selectbox("Vainas de fruta", ["none", "few", "some", "many"])
    fruit_spots = st.selectbox("Manchas en fruta", ["yes", "no"])
    seed = st.selectbox("Semillas", ["normal", "abnormal"])
    mold_growth = st.selectbox("Crecimiento de moho", ["yes", "no"])
    seed_discolor = st.selectbox("Descoloración de semilla", ["yes", "no"])
    seed_size = st.selectbox("Tamaño de semilla", ["small", "medium", "large"])
    shriveling = st.selectbox("Arrugamiento", ["yes", "no"])
    roots = st.selectbox("Raíces", ["normal", "abnormal"])

    submitted = st.form_submit_button("Diagnosticar")

if submitted:
    try:
        # Crear dataframe con los datos ingresados
        input_data = pd.DataFrame({
            'date': [date],
            'plant-stand': [plant_stand],
            'precip': [precip],
            'temp': [temp],
            'hail': [hail],
            'crop-hist': [crop_hist],
            'area-damaged': [area_damaged],
            'severity': [severity],
            'seed-tmt': [seed_tmt],
            'germination': [germination],
            'plant-growth': [plant_growth],
            'leaves': [leaves],
            'leafspots-halo': [leafspots_halo],
            'leafspots-marg': [leafspots_marg],
            'leafspot-size': [leafspot_size],
            'leaf-shread': [leaf_shread],
            'leaf-malf': [leaf_malf],
            'leaf-mild': [leaf_mild],
            'stem': [stem],
            'lodging': [lodging],
            'stem-cankers': [stem_cankers],
            'canker-lesion': [canker_lesion],
            'fruiting-bodies': [fruiting_bodies],
            'external-decay': [external_decay],
            'mycelium': [mycelium],
            'int-discolor': [int_discolor],
            'sclerotia': [sclerotia],
            'fruit-pods': [fruit_pods],
            'fruit-spots': [fruit_spots],
            'seed': [seed],
            'mold-growth': [mold_growth],
            'seed-discolor': [seed_discolor],
            'seed-size': [seed_size],
            'shriveling': [shriveling],
            'roots': [roots],
        })

        # Asegurarse de que las columnas coincidan con las que se entrenaron
        # Obtenemos las columnas esperadas del encoder
        expected_columns = encoder.get_feature_names_out(input_data.columns)
        
        # Reindexar las columnas del DataFrame para alinearlas con las esperadas
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)

        # Preprocesamiento
        X_encoded = encoder.transform(input_data)
        X_scaled = scaler.transform(X_encoded)
        
        # Predicción
        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled).max()
        
        # Mostrar resultados
        st.success(f"Diagnóstico: {prediction[0]} (Confianza: {probability:.2%})")
        
    except Exception as e:
        st.error(f"Error al realizar la predicción: {str(e)}")