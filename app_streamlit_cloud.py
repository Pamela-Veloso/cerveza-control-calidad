"""
APLICACIÓN STREAMLIT - SISTEMA INTELIGENTE PARA CERVECERÍAS
VERSIÓN CLOUD (ONNX) - OPTIMIZADA
Interfaz web para usar los 3 modelos de Deep Learning
TEMA: CERVECERÍA ARTESANAL
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json

# Import ONNX para Cloud
import onnxruntime as ort

import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="Sistema Inteligente Cervecero",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO - TEMA CERVECERO 
st.markdown("""
    <style>
    /* FONDO PRINCIPAL - BLANCO/CREMA */
    .main {
        background: #FFFFFF;
        color: #2C1810;
    }
    
    /* ELEMENTOS DENTRO DEL MAIN - BLANCOS */
    .stMarkdown, .element-container, .stExpander {
        background-color: transparent !important;
        color: #2C1810 !important;
    }
    
    /* CONTENIDO DEL EXPANDER - BLANCO */
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        color: #2C1810 !important;
        border: 2px solid #D4741D;
        border-radius: 8px;
        padding: 20px;
    }
    
    /* SIDEBAR - OSCURO */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2C1810 0%, #1A1A1A 100%);
        border-right: 3px solid #D4741D;
    }
    
    [data-testid="stSidebar"] * {
        color: #F4E4C1 !important;
    }
    
    /* TÍTULOS */
    h1 {
        color: #2C1810 !important;
        text-shadow: 2px 2px 4px rgba(212, 116, 29, 0.2);
    }
    
    h2, h3 {
        color: #D4741D !important;
        text-shadow: none;
    }
    
    /* BOTONES */
    .stButton>button {
        background: linear-gradient(135deg, #D4741D 0%, #F4A950 100%);
        color: #FFFFFF;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 30px;
        border: 2px solid #F4A950;
        box-shadow: 0 4px 8px rgba(212, 116, 29, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(244, 169, 80, 0.5);
        background: linear-gradient(135deg, #F4A950 0%, #D4741D 100%);
    }
    
    /* CAJA DE PREDICCIÓN */
    .prediction-box {
        background: linear-gradient(135deg, #D4741D 0%, #F4A950 100%);
        color: #FFFFFF;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        margin: 25px 0;
        box-shadow: 0 8px 16px rgba(212, 116, 29, 0.4);
        border: 3px solid #F4A950;
    }
    
    /* TARJETAS DE MÉTRICAS - BLANCAS */
    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #D4741D;
        box-shadow: 0 4px 8px rgba(212, 116, 29, 0.2);
        margin: 12px 0;
    }
    
    .metric-title {
        color: #D4741D;
        font-weight: bold;
        font-size: 15px;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #2C1810;
    }
    
    /* INPUTS - BLANCOS */
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: #2C1810 !important;
        border: 2px solid #D4741D !important;
        border-radius: 8px;
    }
    
    /* EXPANDER HEADER - CLARO */
    .streamlit-expanderHeader {
        background-color: #FFF8DC !important;
        color: #2C1810 !important;
        border-radius: 8px;
        border: 2px solid #D4741D;
        padding: 10px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #FFE4B5 !important;
    }
    
    /* EXPANDER HEADER SVG (icono) */
    .streamlit-expanderHeader svg {
        fill: #D4741D !important;
    }
    
    /* INFO/SUCCESS/WARNING/ERROR BOXES - ÁREA PRINCIPAL */
    .stAlert, [data-testid="stNotification"] {
        background-color: #FFFFFF !important;
        color: #2C1810 !important;
        border: 2px solid #D4741D !important;
        border-radius: 8px;
    }

    .stSuccess {
        background-color: #F1F8F4 !important;
        border-left: 5px solid #4CAF50 !important;
        color: #2C1810 !important;
    }

    .stWarning {
        background-color: #FFF8F0 !important;
        border-left: 5px solid #FF9800 !important;
        color: #2C1810 !important;
    }

    .stError {
        background-color: #FFF5F5 !important;
        border-left: 5px solid #F44336 !important;
        color: #2C1810 !important;
    }

    .stInfo {
        background-color: #F0F8FF !important;
        border-left: 5px solid #2196F3 !important;
        color: #2C1810 !important;
    }

    /* INFO BOXES ESPECÍFICAMENTE EN SIDEBAR */
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] .stInfo {
        background-color: #FFF8DC !important;
        border: 2px solid #F4A950 !important;
        border-left: 5px solid #F4A950 !important;
    }

    /* FORZAR TEXTO OSCURO EN INFO BOXES DEL SIDEBAR */
    [data-testid="stSidebar"] .stAlert *,
    [data-testid="stSidebar"] .stInfo * {
        color: #2C1810 !important;
    }
    
    /* MÉTRICAS DE STREAMLIT */
    [data-testid="stMetricValue"] {
        color: #D4741D !important;
        font-size: 28px !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2C1810 !important;
    }
    
    /* TABLAS */
    .dataframe {
        background-color: #FFFFFF !important;
        color: #2C1810 !important;
    }
    
    /* SEPARADOR */
    hr {
        border-color: #D4741D !important;
        opacity: 0.5;
    }
    
    /* SPINNER */
    .stSpinner > div {
        border-top-color: #D4741D !important;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 10px;
        background: #F5F5F5;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #D4741D 0%, #F4A950 100%);
        border-radius: 5px;
    }
    
    /* RADIO BUTTONS EN SIDEBAR */
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(244, 169, 80, 0.1);
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #F4A950;
        margin: 5px 0;
    }
    
    /* TEXTO GENERAL - NEGRO */
    p, span, label, div, li, code {
        color: #2C1810 !important;
    }
    
    /* EXCEPTO EN SIDEBAR */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #F4E4C1 !important;
    }
    
    /* HEADER DECORACIÓN */
    .header-decoration {
        background: linear-gradient(90deg, 
            transparent 0%, 
            #D4741D 20%, 
            #F4A950 50%, 
            #D4741D 80%, 
            transparent 100%);
        height: 3px;
        margin: 20px 0;
        border-radius: 2px;
    }
    
    /* SUBHEADERS */
    .stSubheader {
        color: #D4741D !important;
    }
    
    /* CAPTION */
    .stCaptionContainer {
        color: #666666 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# CARGAR MODELOS ONNX, ENCODERS Y MÉTRICAS
# ============================================

@st.cache_resource
def cargar_todo():
    """Carga modelos ONNX, encoders y métricas (se cachea para eficiencia)"""
    
    # Modelos ONNX
    modelo_1 = ort.InferenceSession('modelos/modelo_1.onnx')
    modelo_2 = ort.InferenceSession('modelos/modelo_2.onnx')
    modelo_3 = ort.InferenceSession('modelos/modelo_3.onnx')
    
    # Encoders
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Métricas
    with open('modelos/modelo_1_metricas.json', 'r') as f:
        metricas_1 = json.load(f)
    
    with open('modelos/modelo_2_metricas.json', 'r') as f:
        metricas_2 = json.load(f)
    
    with open('modelos/modelo_3_metricas.json', 'r') as f:
        metricas_3 = json.load(f)
    
    return modelo_1, modelo_2, modelo_3, scaler, label_encoder, metricas_1, metricas_2, metricas_3

# Cargar todo
try:
    modelo_1, modelo_2, modelo_3, scaler, label_encoder, metricas_1, metricas_2, metricas_3 = cargar_todo()
    modelos_cargados = True
except Exception as e:
    st.error(f"❌ Error al cargar modelos: {e}")
    st.stop()

# ============================================
# HEADER CON DECORACIÓN
# ============================================

st.title("🍺 Sistema Inteligente para Cervecerías Artesanales")
st.markdown('<div class="header-decoration"></div>', unsafe_allow_html=True)
st.markdown("**Sistema Multi-Modelo de Deep Learning para Control de Calidad Cervecero**")
st.markdown("---")

# Información del sistema
with st.expander("ℹ️ Acerca de este sistema", expanded=False):
    st.markdown("""
    ### 🎯 Sistema Multi-Modelo de Deep Learning
    
    Este sistema utiliza **3 modelos de redes neuronales** independientes para resolver 
    diferentes problemas de las cervecerías artesanales:
    
    - **Modelo 1: Control de Calidad** - Clasifica el estilo de cerveza
    - **Modelo 2: Predictor de ABV** - Estima el contenido de alcohol
    - **Modelo 3: Clasificador Experimental** - Analiza recetas experimentales
    
    **Tecnología:** TensorFlow/Keras, Deep Learning, Data Augmentation (SMOTE)
    
    **Dataset:** 150 muestras originales → 253 muestras con augmentation
    
    **Desarrollado para:** Cervecerías Artesanales de Valdivia, Chile
    """)

# ============================================
# SIDEBAR - SELECCIÓN DE MODELO
# ============================================

st.sidebar.title("🍺 MAESTRO CERVECERO")
st.sidebar.markdown("---")

st.sidebar.subheader("¿Qué necesitas hacer hoy?")

modelo_seleccionado = st.sidebar.radio(
    "Selecciona una opción:",
    [
        "🎯 Modelo 1: Control de Calidad",
        "📈 Modelo 2: Predecir ABV",
        "🔬 Modelo 3: Clasificador Recetas"
    ],
    index=0
)

st.sidebar.markdown("---")

# Mostrar métricas del modelo seleccionado en el sidebar
st.sidebar.subheader("📊 Rendimiento del Modelo")

if "Modelo 1" in modelo_seleccionado:
    st.sidebar.metric(
        label="Accuracy",
        value=f"{metricas_1['accuracy']*100:.2f}%",
        delta="Excelente" if metricas_1['accuracy'] > 0.95 else "Bueno"
    )
    st.sidebar.caption(f"✅ Entrenado con {metricas_1['epochs_trained']} epochs")
    st.sidebar.progress(metricas_1['accuracy'])
    
elif "Modelo 2" in modelo_seleccionado:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric(label="R²", value=f"{metricas_2['r2']:.3f}")
    with col2:
        st.metric(label="RMSE", value=f"{metricas_2['rmse']:.3f}")
    
    st.sidebar.metric(label="MAE", value=f"{metricas_2['mae']:.4f}%")
    st.sidebar.caption(f"✅ Entrenado con {metricas_2['epochs_trained']} epochs")
    
    r2_normalized = max(0, min(1, metricas_2['r2']))
    st.sidebar.progress(r2_normalized)
    
else:  # Modelo 3
    st.sidebar.metric(
        label="Accuracy",
        value=f"{metricas_3['accuracy']*100:.2f}%",
        delta="Excelente" if metricas_3['accuracy'] > 0.95 else "Bueno"
    )
    st.sidebar.caption(f"✅ Entrenado con {metricas_3['epochs_trained']} epochs")
    st.sidebar.progress(metricas_3['accuracy'])

st.sidebar.markdown("---")
st.sidebar.info("""
📚 **Instrucciones:**
1. Selecciona el modelo según tu necesidad
2. Ingresa los parámetros de tu lote
3. Presiona "Analizar"
4. ¡Obtén resultados instantáneos!
""")

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def normalizar_inputs(og, abv, ph, ibu):
    """Normaliza los inputs usando el scaler entrenado"""
    abv_og_ratio = abv / og
    ibu_abv_ratio = ibu / abv
    features = np.array([[og, abv, ph, ibu, abv_og_ratio, ibu_abv_ratio]])
    features_norm = scaler.transform(features)
    return features_norm

def predecir_onnx(modelo, features):
    """Ejecuta predicción con modelo ONNX"""
    input_name = modelo.get_inputs()[0].name
    return modelo.run(None, {input_name: features.astype(np.float32)})[0]

def crear_gauge_chart(valor, titulo, rango_min, rango_max):
    """Crea un gráfico gauge con tema cervecero - COLORES VISIBLES"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={'text': titulo, 'font': {'color': '#2C1810', 'size': 24, 'weight': 'bold'}},
        number={'font': {'color': '#2C1810', 'size': 48, 'weight': 'bold'}},
        gauge={
            'axis': {
                'range': [rango_min, rango_max], 
                'tickcolor': '#2C1810', 
                'tickfont': {'color': '#2C1810', 'size': 14, 'weight': 'bold'}
            },
            'bar': {'color': '#D4741D', 'thickness': 0.8},
            'bgcolor': '#FFF8DC',
            'borderwidth': 3,
            'bordercolor': '#D4741D',
            'steps': [
                {'range': [rango_min, rango_min + (rango_max-rango_min)*0.33], 'color': "#FFE4B5"},
                {'range': [rango_min + (rango_max-rango_min)*0.33, rango_min + (rango_max-rango_min)*0.66], 'color': "#FFDAB9"},
                {'range': [rango_min + (rango_max-rango_min)*0.66, rango_max], 'color': "#F4C2A0"}
            ],
            'threshold': {
                'line': {'color': "#8B4513", 'width': 5},
                'thickness': 0.85,
                'value': valor
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        font={'color': '#2C1810', 'size': 14},
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def mostrar_metricas_modelo(metricas, modelo_num):
    """Muestra las métricas de rendimiento del modelo"""
    st.markdown("---")
    st.subheader("📊 Confiabilidad del Modelo")
    
    if modelo_num in [1, 3]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">ACCURACY</div>
                    <div class="metric-value">{metricas['accuracy']*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">EPOCHS ENTRENADOS</div>
                    <div class="metric-value">{metricas['epochs_trained']}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            calidad = "🟢 Excelente" if metricas['accuracy'] > 0.95 else "🟡 Bueno" if metricas['accuracy'] > 0.85 else "🟠 Aceptable"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">CALIFICACIÓN</div>
                    <div class="metric-value" style="font-size: 20px;">{calidad}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.info(f"""
        **Interpretación:** El modelo tiene una precisión del **{metricas['accuracy']*100:.2f}%**, 
        lo que significa que clasifica correctamente **{int(metricas['accuracy']*100)} de cada 100 lotes**.
        """)
        
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">R² SCORE</div>
                    <div class="metric-value">{metricas['r2']:.3f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">RMSE</div>
                    <div class="metric-value">{metricas['rmse']:.3f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">MAE</div>
                    <div class="metric-value">{metricas['mae']:.3f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            calidad = "🟢 Excelente" if metricas['r2'] > 0.8 else "🟡 Bueno" if metricas['r2'] > 0.6 else "🟠 Aceptable"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">CALIFICACIÓN</div>
                    <div class="metric-value" style="font-size: 20px;">{calidad}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.info(f"""
        **Interpretación:** 
        - **R² = {metricas['r2']:.3f}:** El modelo explica el {metricas['r2']*100:.1f}% de la varianza en ABV
        - **MAE = {metricas['mae']:.3f}%:** Error promedio de ±{metricas['mae']:.2f}% en las predicciones
        - **RMSE = {metricas['rmse']:.3f}%:** Raíz del error cuadrático medio
        """)

# ============================================
# MODELO 1: CONTROL DE CALIDAD
# ============================================

if "Modelo 1" in modelo_seleccionado:
    st.header("🎯 Modelo 1: Control de Calidad")
    
    st.markdown("""
    ### ❓ ¿Para qué sirve?
    Este modelo **clasifica el estilo de cerveza** basándose en sus parámetros fisicoquímicos.
    
    **Pregunta que responde:** *"¿Este lote salió bien? ¿Corresponde al estilo que quería hacer?"*
    """)
    
    mostrar_metricas_modelo(metricas_1, 1)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Parámetros del Lote")
        og = st.number_input("**OG** (Original Gravity) [°Plato]", 
                            min_value=8.0, max_value=16.0, value=12.0, step=0.1,
                            help="Densidad inicial del mosto")
        abv = st.number_input("**ABV** (Alcohol by Volume) [%]", 
                             min_value=2.0, max_value=8.0, value=5.0, step=0.1,
                             help="Porcentaje de alcohol")
    
    with col2:
        st.write("")
        st.write("")
        ph = st.number_input("**pH** (Acidez)", 
                            min_value=3.0, max_value=5.0, value=4.2, step=0.1,
                            help="Nivel de acidez")
        ibu = st.number_input("**IBU** (Amargor)", 
                             min_value=5, max_value=60, value=20, step=1,
                             help="Unidades de amargor")
    
    if st.button("🔍 Analizar Lote", key="btn_modelo1"):
        with st.spinner("Analizando..."):
            features_norm = normalizar_inputs(og, abv, ph, ibu)
            prediccion = predecir_onnx(modelo_1, features_norm)
            clase_predicha = np.argmax(prediccion[0])
            estilo = label_encoder.classes_[clase_predicha]
            confianza = prediccion[0][clase_predicha] * 100
            
            st.markdown(f"""
                <div class="prediction-box">
                    🍺 ESTILO DETECTADO: {estilo}<br>
                    Confianza: {confianza:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Probabilidades por Estilo")
                
                df_prob = pd.DataFrame({
                    'Estilo': label_encoder.classes_,
                    'Probabilidad (%)': prediccion[0] * 100
                })
                
                # Gráfico limpio sin texto en barras
                fig = px.bar(df_prob, x='Estilo', y='Probabilidad (%)',
                            color='Probabilidad (%)',
                            color_continuous_scale=[[0, '#8B4513'], [0.5, '#D4741D'], [1, '#F4A950']])
                
                fig.update_traces(
                    texttemplate='',
                    hovertemplate='<b>%{x}</b><br>Probabilidad: %{y:.1f}%<extra></extra>'
                )
                
                fig.update_layout(
                    showlegend=False, 
                    height=400,
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FFFFFF',
                    font=dict(color='#2C1810', size=14),
                    xaxis=dict(
                        title='Estilo de Cerveza',
                        title_font=dict(size=16, color='#2C1810', weight='bold'),
                        tickfont=dict(size=14, color='#2C1810', weight='bold')
                    ),
                    yaxis=dict(
                        title='Probabilidad (%)',
                        title_font=dict(size=16, color='#2C1810', weight='bold'),
                        gridcolor='#E0E0E0',
                        tickfont=dict(size=14, color='#2C1810', weight='bold'),
                        range=[0, 105]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Detalles de la Predicción")
                
                for i, style in enumerate(label_encoder.classes_):
                    prob = prediccion[0][i] * 100
                    st.metric(label=style, value=f"{prob:.2f}%")
                
                st.success("✅ **Interpretación:**")
                if confianza > 90:
                    st.write(f"El lote corresponde claramente a una **{estilo}**.")
                elif confianza > 70:
                    st.write(f"El lote es probablemente una **{estilo}**, con buena confianza.")
                else:
                    st.warning(f"El lote podría ser una **{estilo}**, pero hay incertidumbre. Revisar parámetros.")

# ============================================
# MODELO 2: PREDICTOR DE ABV
# ============================================

elif "Modelo 2" in modelo_seleccionado:
    st.header("📈 Modelo 2: Predictor de ABV")
    
    st.markdown("""
    ### ❓ ¿Para qué sirve?
    Este modelo **predice el contenido de alcohol (ABV)** basándose en la densidad inicial (OG), amargor (IBU) y pH.
    
    **Pregunta que responde:** *"¿Cuánto alcohol tendrá mi cerveza cuando termine de fermentar?"*
    
    **⚠️ Útil ANTES de terminar la fermentación** permite anticipar el comportamiento del lote, detectar desviaciones potenciales y tomar decisiones correctivas antes de completar el proceso.
    """)
    
    mostrar_metricas_modelo(metricas_2, 2)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        og = st.number_input("**OG** (Original Gravity) [°Plato]", 
                            min_value=8.0, max_value=16.0, value=13.0, step=0.1,
                            help="Densidad inicial del mosto")
    
    with col2:
        ph = st.number_input("**pH** (Acidez)", 
                            min_value=3.0, max_value=5.0, value=4.2, step=0.1,
                            help="Nivel de acidez")
    
    with col3:
        ibu = st.number_input("**IBU** (Amargor)", 
                             min_value=5, max_value=60, value=25, step=1,
                             help="Unidades de amargor")
    
    if st.button("🔮 Predecir ABV", key="btn_modelo2"):
        with st.spinner("Calculando..."):
            features_temp = normalizar_inputs(og, 5.0, ph, ibu)
            features_modelo2 = features_temp[:, [0, 2, 3]]
            abv_predicho = predecir_onnx(modelo_2, features_modelo2)[0][0]
            
            st.markdown(f"""
                <div class="prediction-box">
                    🍺 ABV ESTIMADO: {abv_predicho:.2f}%
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Visualización")
                fig = crear_gauge_chart(
                    valor=abv_predicho,
                    titulo="ABV Estimado (%)",
                    rango_min=2.0,
                    rango_max=8.0
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Rango de Estimación")
                mae = metricas_2['mae']
                
                st.metric(
                    label="ABV Estimado",
                    value=f"{abv_predicho:.2f}%",
                    delta=f"± {mae:.2f}%"
                )
                
                st.info(f"""
                **Rango probable:**  
                {abv_predicho - mae:.2f}% - {abv_predicho + mae:.2f}%
                
                *Basado en MAE del modelo: {mae:.3f}%*
                """)
                
                st.success("✅ **Interpretación:**")
                if abv_predicho < 3.5:
                    st.write("Cerveza ligera, tipo Light Lager.")
                elif abv_predicho < 5.5:
                    st.write("Contenido alcohólico típico de Lagers premium.")
                else:
                    st.write("Alto contenido alcohólico, tipo IPA o cerveza fuerte.")

# ============================================
# MODELO 3: CLASIFICADOR EXPERIMENTAL
# ============================================

elif "Modelo 3" in modelo_seleccionado:
    st.header("🔬 Modelo 3: Clasificador Experimental")
    
    st.markdown("""
    ### ❓ ¿Para qué sirve?
    Este modelo **analiza recetas experimentales** y proporciona un análisis probabilístico detallado.
    
    **Pregunta que responde:** *"¿A qué estilo se parece más mi nueva receta experimental?"*
    
    **💡 Ideal para:** Desarrollo de nuevos productos, innovación, pruebas de mercado.
    """)
    
    mostrar_metricas_modelo(metricas_3, 3)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Parámetros de la Receta Experimental")
        og = st.number_input("**OG** (Original Gravity) [°Plato]", 
                            min_value=8.0, max_value=16.0, value=11.5, step=0.1)
        abv = st.number_input("**ABV** (Alcohol by Volume) [%]", 
                             min_value=2.0, max_value=8.0, value=5.5, step=0.1)
    
    with col2:
        st.write("")
        st.write("")
        ph = st.number_input("**pH** (Acidez)", 
                            min_value=3.0, max_value=5.0, value=4.0, step=0.1)
        ibu = st.number_input("**IBU** (Amargor)", 
                             min_value=5, max_value=60, value=30, step=1)
    
    if st.button("🔬 Analizar Receta Experimental", key="btn_modelo3"):
        with st.spinner("Analizando perfil de la receta..."):
            features_norm = normalizar_inputs(og, abv, ph, ibu)
            prediccion = predecir_onnx(modelo_3, features_norm)
            clase_predicha = np.argmax(prediccion[0])
            estilo_principal = label_encoder.classes_[clase_predicha]
            confianza_principal = prediccion[0][clase_predicha] * 100
            
            st.markdown(f"""
                <div class="prediction-box">
                    🔬 ESTILO MÁS CERCANO: {estilo_principal}<br>
                    Similaridad: {confianza_principal:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Análisis Probabilístico Completo")
            
            # Colores bien diferenciados
            colores_distintos = ['#D4741D', '#8B4513', '#FFB347']
            
            fig = go.Figure(data=[go.Pie(
                labels=label_encoder.classes_,
                values=prediccion[0] * 100,
                hole=.3,
                marker=dict(
                    colors=colores_distintos,
                    line=dict(color='#FFFFFF', width=3)
                ),
                textinfo='label+percent',
                textfont=dict(size=18, color='#FFFFFF', weight='bold'),
                hovertemplate='<b>%{label}</b><br>Similaridad: %{value:.1f}%<extra></extra>',
                pull=[0.05, 0.05, 0.05]
            )])
            
            fig.update_layout(
                title=dict(
                    text="Distribución de Similaridad por Estilo",
                    font=dict(size=20, color='#2C1810', weight='bold')
                ),
                height=500,
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF',
                font=dict(color='#2C1810', size=14),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.1,
                    font=dict(size=16, color='#2C1810', weight='bold'),
                    bgcolor='#FFF8DC',
                    bordercolor='#D4741D',
                    borderwidth=2
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Detalle por Estilo")

            df_analisis = pd.DataFrame({
                'Estilo': label_encoder.classes_,
                'Similaridad (%)': prediccion[0] * 100
            }).sort_values('Similaridad (%)', ascending=False).reset_index(drop=True)

            # Agregar ranking
            rankings = ['🥇 Primero', '🥈 Segundo', '🥉 Tercero']
            df_analisis.insert(0, 'Posición', rankings[:len(df_analisis)])

            # Colorear con NEGRO para alto contraste
            def colorear_tabla(row):
                val = row['Similaridad (%)']
    
                if val > 70:
                    return [
                        'background-color: #C85A17; color: #FFFFFF; font-weight: bold; font-size: 20px; text-align: center; padding: 20px; border: 3px solid #000000;',
                        'background-color: #C85A17; color: #FFFFFF; font-weight: bold; font-size: 20px; text-align: left; padding: 20px; border: 3px solid #000000;',
                        'background-color: #C85A17; color: #FFFFFF; font-weight: bold; font-size: 28px; text-align: right; padding: 20px; border: 3px solid #000000;'
            ]
                elif val > 30:
                    return [
                        'background-color: #F4A950; color: #000000; font-weight: bold; font-size: 20px; text-align: center; padding: 20px; border: 3px solid #000000;',
                        'background-color: #F4A950; color: #000000; font-weight: bold; font-size: 20px; text-align: left; padding: 20px; border: 3px solid #000000;',
                        'background-color: #F4A950; color: #000000; font-weight: bold; font-size: 28px; text-align: right; padding: 20px; border: 3px solid #000000;'
            ]
                else:
                    return [
                        'background-color: #FFD89C; color: #000000; font-weight: bold; font-size: 20px; text-align: center; padding: 20px; border: 3px solid #000000;',
                        'background-color: #FFD89C; color: #000000; font-weight: bold; font-size: 20px; text-align: left; padding: 20px; border: 3px solid #000000;',
                        'background-color: #FFD89C; color: #000000; font-weight: bold; font-size: 28px; text-align: right; padding: 20px; border: 3px solid #000000;'
            ]

            st.dataframe(
                df_analisis.style.apply(colorear_tabla, axis=1).format({'Similaridad (%)': '{:.2f}%'}),
                use_container_width=True,
                height=240,
                hide_index=True
            )
            
            st.success("✅ **Recomendaciones:**")
            
            probs_sorted = sorted(zip(label_encoder.classes_, prediccion[0]), 
                                 key=lambda x: x[1], reverse=True)
            
            if probs_sorted[0][1] > 0.8:
                st.write(f"✅ Tu receta es claramente una **{probs_sorted[0][0]}**. "
                        "Puedes comercializarla con ese nombre.")
            elif probs_sorted[0][1] > 0.6:
                st.write(f"✅ Tu receta se parece más a una **{probs_sorted[0][0]}**, "
                        "aunque tiene características de otros estilos.")
            else:
                st.write(f"🎨 Tu receta es **híbrida**, con características de:")
                for style, prob in probs_sorted:
                    if prob > 0.2:
                        st.write(f"  • {style}: {prob*100:.1f}%")
                st.write("\n💡 Considera comercializarla como una **cerveza experimental** "
                        "o crear una nueva categoría.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown('<div class="header-decoration"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #2C1810;'>
    <p style='font-size: 18px; font-weight: bold; color: #D4741D;'>🍺 Sistema Inteligente para Cervecerías Artesanales</p>
    <p style='color: #2C1810;'>Valdivia, Chile</p>
    <p style='color: #2C1810;'>Desarrollado con Deep Learning (TensorFlow/Keras) • 3 Modelos Independientes</p>
    <p style='color: #2C1810;'>Dataset: 150 muestras → 253 con augmentation | Accuracy: 95%+ | R²: Positivo</p>
    <p style='font-size: 14px; color: #D4741D;'>Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)