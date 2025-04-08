# Segundoparcial_DataSience
## Autor
Danna Corral 

## Descripción del Proyecto
Este proyecto es un sistema de diagnóstico para enfermedades en plantas de soja, diseñado como parte del segundo parcial de una asignatura de Ciencia de Datos. Utiliza un modelo de SVM (Support Vector Machine) para clasificar enfermedades basándose en diversas características observadas en las plantas. La aplicación incluye:

- Entrenamiento del modelo a partir de un dataset de la UCI Machine Learning Repository.
- Preprocesamiento de datos que incluye codificación categórica, escalado y balanceo de clases.
- Una interfaz interactiva desarrollada con Streamlit para facilitar el diagnóstico.

## Características Principales
- **Modelo de Clasificación**: Entrenado con datos de plantas de soja, utilizando SVM optimizado con hiperparámetros.
- **Preprocesamiento Avanzado**: Incluye el uso de OneHotEncoder, StandardScaler y SMOTE para preparar los datos antes del entrenamiento.
- **Interfaz de Usuario**: Diseñada con Streamlit, permite ingresar síntomas de manera interactiva y obtener un diagnóstico con confianza.

## Requisitos del Sistema

### Dependencias
Asegúrate de tener instalado Python 3.8 o superior y las siguientes bibliotecas:

- pandas
- joblib
- scikit-learn
- imbalanced-learn
- requests
- streamlit

Puedes instalarlas usando el siguiente comando:

```bash
pip install -r requirements.txt
```

### Archivos Necesarios
- `modelo.py`: Contiene el código para entrenar y cargar el modelo.
- `app.py`: La aplicación Streamlit para la interfaz de usuario.
- Dataset: El script descarga automáticamente el dataset desde la [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data).

## Ejecución del Proyecto

### Entrenamiento del Modelo
Si no existe un modelo entrenado (`svm_model_soybean.pkl`), el script `modelo.py` entrenará uno nuevo:

```bash
python modelo.py
```

### Ejecución de la Aplicación
Para iniciar la interfaz interactiva de diagnóstico, ejecuta el archivo `app.py` con Streamlit:

```bash
streamlit run app.py
```

Esto abrirá la aplicación en tu navegador web.

## Estructura del Proyecto

```
Segundoparcial_DataSience/
├── modelo.py
├── app.py
├── requirements.txt
├── .gitignore
├── svm_model_soybean.pkl  # Archivo generado tras entrenar el modelo
└── README.md
```

## Archivo `.gitignore`
El archivo `.gitignore` incluye configuraciones para ignorar archivos innecesarios, como:
- Archivos generados por Python (`*.pyc`, `__pycache__/`).
- Modelos y datasets generados (`svm_model_soybean.pkl`, `soybean-small.data`).
- Directorios de entornos virtuales (`env/`, `venv/`).

## Notas
- El modelo está configurado para diagnosticar solo las 4 enfermedades más frecuentes en el dataset.
- El dataset debe descargarse desde la URL especificada en el script, o bien debe estar disponible localmente.

## Contribuciones
Este proyecto fue desarrollado como un ejercicio académico. Si tienes sugerencias o mejoras, no dudes en contribuir abriendo un pull request.

## Licencia
Este proyecto es de uso educativo y no tiene una licencia específica. Consulta con el instructor para más información sobre su uso.

