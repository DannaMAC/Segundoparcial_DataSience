import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import requests
from io import StringIO
from collections import Counter

def cargar_modelo():
    """Carga el modelo entrenado, el escalador y el codificador"""
    try:
        model, scaler, encoder = joblib.load('svm_model_soybean.pkl')
        return model, scaler, encoder
    except FileNotFoundError:
        print("Modelo no encontrado. Entrenando nuevo modelo...")
        return entrenar_modelo()

def entrenar_modelo():
    """Función principal para entrenar el modelo"""
    # Cargar datos desde la URL correcta
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data"
    column_names = ['class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 
                   'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 
                   'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem', 
                   'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external-decay', 
                   'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit-spots', 'seed', 
                   'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']

    try:
        # Descargar datos desde la URL
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data, header=None, names=column_names)
            print("Datos cargados exitosamente desde la URL.")
        else:
            raise Exception(f"Error al cargar datos. Código: {response.status_code}")
    except Exception as e:
        print(e)
        try:
            df = pd.read_csv('soybean-small.data', header=None, names=column_names)
            print("Datos cargados desde archivo local.")
        except FileNotFoundError:
            print("No se pudo cargar el dataset.")
            return None, None, None

    # Limpieza de datos
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    print(f"\nFilas después de limpieza: {len(df)}")

    # Reducir a las 4 clases principales
    top_classes = df['class'].value_counts().nlargest(4).index
    df = df[df['class'].isin(top_classes)]
    print("\nDistribución después de filtrar clases:")
    print(df['class'].value_counts())

    # Preprocesamiento
    categorical_columns = column_names[1:]  # Todas excepto 'class'
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded = encoder.fit_transform(df[categorical_columns])
    X = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    y = df['class']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Escalado (importante para SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balanceo de clases con SMOTE ajustado dinámicamente
    min_samples = Counter(y_train).most_common()[-1][1]  # Muestras en la clase más pequeña
    k_neighbors = min(2, min_samples - 1)

    if k_neighbors < 1:
        print("\nUsando RandomOverSampler por clases muy pequeñas")
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
    else:
        print(f"\nUsando SMOTE con k_neighbors={k_neighbors}")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Configuración del modelo SVM optimizado
    best_params = {
        'C': 10,
        'gamma': 'scale',
        'kernel': 'rbf',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42
    }
    
    model = SVC(**best_params)
    model.fit(X_train_resampled, y_train_resampled)

    # Evaluación
    y_pred = model.predict(X_test_scaled)
    print("\nInforme de clasificación:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

    # Guardar modelo
    joblib.dump((model, scaler, encoder), 'svm_model_soybean.pkl')
    print("\nModelo SVM guardado exitosamente.")
    
    return model, scaler, encoder

if __name__ == "__main__":
    model, scaler, encoder = entrenar_modelo()
