# Sistema Inteligente para Cervecerías Artesanales

Sistema multi-modelo de Deep Learning para control de calidad en cervecerías artesanales de Valdivia, Chile.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_DE_TU_APP)

---

## Descripción

Este proyecto utiliza **3 modelos de redes neuronales** independientes para resolver diferentes problemas operativos de las cervecerías artesanales:

### Modelo 1: Control de Calidad
- **Objetivo:** Clasificar el estilo de cerveza basándose en parámetros fisicoquímicos
- **Arquitectura:** Red neuronal densa (6 → 128 → 64 → 32 → 3)
- **Accuracy:** 95.65%
- **Pregunta que resuelve:** *"¿Este lote salió bien? ¿Corresponde al estilo que quería hacer?"*

### Modelo 2: Predictor de ABV
- **Objetivo:** Predecir el contenido de alcohol (ABV) antes de terminar la fermentación
- **Arquitectura:** Red neuronal densa (3 → 64 → 32 → 16 → 1)
- **R²:** Positivo (ver métricas completas en reportes)
- **Pregunta que resuelve:** *"¿Cuánto alcohol tendrá mi cerveza cuando termine de fermentar?"*

### Modelo 3: Clasificador Experimental
- **Objetivo:** Analizar recetas experimentales con análisis probabilístico
- **Arquitectura:** Red neuronal densa (6 → 96 → 48 → 24 → 3)
- **Accuracy:** 95%+
- **Pregunta que resuelve:** *"¿A qué estilo se parece más mi nueva receta experimental?"*

---

## Tecnologías

- **Deep Learning:** TensorFlow/Keras
- **Data Science:** NumPy, Pandas, Scikit-learn
- **Visualización:** Plotly, Matplotlib, Seaborn
- **Web App:** Streamlit
- **Data Augmentation:** SMOTE (150 → 253 muestras)

---

## Dataset

- **Origen:** Ankur's Beer Data Set (https://www.kaggle.com/datasets/ankurnapa/ankurs-beer-data-set) 
- **Muestras originales:** 150
- **Muestras con augmentation:** 253
- **Features:** OG, ABV, pH, IBU + 2 derivadas
- **Estilos:** Premium Lager, IPA, Light Lager

---

## Instalación

### Requisitos Previos
- Python 3.9+
- pip

### Clonar el Repositorio
```bash
git clone https://github.com/Pamela-Veloso/cerveza-control-calidad.git
cd cerveza-control-calidad
```

### Crear Entorno Virtual
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

### Instalar Dependencias
```bash
pip install -r requirements.txt
```

---

## Uso

### Ejecutar Aplicación Web
```bash
streamlit run app_streamlit.py
```

La app se abrirá en `http://localhost:8501`

### Entrenar Modelos (opcional)
```bash
# 1. Exploración de datos
python exploracion_datos.py

# 2. Preparación de datos
python preparacion_datos.py

# 3. Entrenar Modelo 1
python modelo_1_control_calidad.py

# 4. Entrenar Modelo 2
python modelo_2_predictor_abv.py

# 5. Entrenar Modelo 3
python modelo_3_clasificador_experimental.py

# 6. Evaluación consolidada
python evaluacion_modelos.py
```

---

## Estructura del Proyecto
```
cerveza-control-calidad/
├── beer.csv                              # Dataset original
├── label_encoder.pkl                     # Codificador de estilos
├── scaler.pkl                            # Normalizador
├── data/                                 # Datos procesados
├── modelos/                              # Modelos entrenados (.h5)
├── exploracion_datos.py               # EDA
├── preparacion_datos.py               # Preprocesamiento
├── modelo_1_control_calidad.py        # Entrenamiento Modelo 1
├── modelo_2_predictor_abv.py          # Entrenamiento Modelo 2
├── modelo_3_clasificador_experimental.py  # Entrenamiento Modelo 3
├── evaluacion_modelos.py          # Evaluación
├── app_streamlit.py                      # Aplicación Web
├── REPORTE_*.html                        # Reportes de resultados
├── requirements.txt                      # Dependencias
└── README.md                             # Este archivo
```

---

## Resultados

### Modelo 1: Control de Calidad
- **Accuracy:** 95.65%
- **Epochs:** 64 (Early Stopping)

### Modelo 2: Predictor de ABV
- **R²:** Positivo
- **MAE:** Bajo error promedio
- **Epochs:** Variable según convergencia

### Modelo 3: Clasificador Experimental
- **Accuracy:** 95%+
- **Análisis probabilístico detallado**

Ver reportes completos en `REPORTE_FINAL_CONSOLIDADO.html`

---

## Valor para Cervecerías

1. ✅ **Automatiza control de calidad** → Reduce errores humanos
2. ✅ **Predicción anticipada de ABV** → Mejor planificación de producción
3. ✅ **Análisis de recetas experimentales** → Facilita innovación
4. ✅ **Sistema completo** → Cubre todo el ciclo productivo

---

## Autores

**Pamela Veloso, Sebastián Saravia, Andrés Torres**  
Estudiante de Ingeniería en Computación - INACAP  
Proyecto: Aplicaciones de Inteligencia Artificial  
Valdivia, Chile  
Noviembre 2025

---

## Licencia

Este proyecto es parte de un trabajo académico.

---

# Requirements.txt
absl-py==2.3.1
altair==5.5.0
astunparse==1.6.3
attrs==25.4.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.11.12
charset-normalizer==3.4.4
click==8.3.0
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
flatbuffers==25.9.23
fonttools==4.60.1
gast==0.6.0
gitdb==4.0.12
GitPython==3.1.45
google-auth==2.43.0
google-auth-oauthlib==1.0.0
google-pasta==0.2.0
grpcio==1.76.0
h5py==3.15.1
idna==3.11
imbalanced-learn==0.11.0
importlib-metadata==6.11.0
Jinja2==3.1.6
joblib==1.5.2
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
keras==2.14.0
kiwisolver==1.4.9
libclang==18.1.1
Markdown==3.10
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.7.2
mdurl==0.1.2
ml-dtypes==0.2.0
narwhals==2.11.0
numpy==1.24.3
oauthlib==3.3.1
opt_einsum==3.4.0
packaging==23.2
pandas==2.0.3
pillow==10.4.0
plotly==5.17.0
protobuf==4.25.8
pyarrow==22.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pydeck==0.9.1
Pygments==2.19.2
pyparsing==3.0.9
python-dateutil==2.9.0.post0
pytz==2025.2
referencing==0.37.0
requests==2.32.5
requests-oauthlib==2.0.0
rich==13.9.4
rpds-py==0.28.0
rsa==4.9.1
scikit-learn==1.3.0
scipy==1.15.3
seaborn==0.12.2
six==1.17.0
smmap==5.0.2
streamlit==1.28.0
tenacity==8.5.0
tensorboard==2.14.1
tensorboard-data-server==0.7.2
tensorflow==2.14.0
tensorflow-estimator==2.14.0
tensorflow-intel==2.14.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor==3.2.0
threadpoolctl==3.6.0
toml==0.10.2
tornado==6.5.2
typing_extensions==4.15.0
tzdata==2025.2
tzlocal==5.3.1
urllib3==2.5.0
validators==0.35.0
watchdog==6.0.0
Werkzeug==3.1.3
wrapt==1.14.2
zipp==3.23.0
