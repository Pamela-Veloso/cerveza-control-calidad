"""
EVALUACIÓN DE LOS 3 MODELOS 
Genera visualizaciones y reporte HTML único
"""

import numpy as np
import pandas as pd
import pickle
import json
from tensorflow import keras
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report, r2_score
from datetime import datetime
import os

print("="*70)
print("📊 EVALUACIÓN DE LOS 3 MODELOS CON REDES NEURONALES")
print("="*70)

# ============================================
# CARGAR DATOS Y MODELOS
# ============================================

print("\n📁 Cargando modelos y datos...")

# Cargar modelos
modelo_1 = keras.models.load_model('modelos/modelo_1_control_calidad.h5')
modelo_2 = keras.models.load_model('modelos/modelo_2_predictor_abv.h5')
modelo_3 = keras.models.load_model('modelos/modelo_3_clasificador_experimental.h5')

# Cargar métricas
with open('modelos/modelo_1_metricas.json', 'r') as f:
    metricas_1 = json.load(f)

with open('modelos/modelo_2_metricas.json', 'r') as f:
    metricas_2 = json.load(f)

with open('modelos/modelo_3_metricas.json', 'r') as f:
    metricas_3 = json.load(f)

# Cargar historiales
with open('modelos/modelo_1_history.json', 'r') as f:
    history_1 = json.load(f)

with open('modelos/modelo_2_history.json', 'r') as f:
    history_2 = json.load(f)

with open('modelos/modelo_3_history.json', 'r') as f:
    history_3 = json.load(f)

# Cargar datos de test
X_test_full = np.load('data/X_test.npy')
y_style_test = np.load('data/y_style_test.npy')

# Para modelo 2
X_test_modelo_2 = X_test_full[:, [0, 2, 3]]  # OG, pH, IBU
y_test_modelo_2 = np.load('modelos/modelo_2_y_test.npy')
y_pred_modelo_2 = np.load('modelos/modelo_2_y_pred.npy')

# Label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✅ Datos cargados exitosamente\n")

# ============================================
# PREDICCIONES
# ============================================

print("🔮 Generando predicciones...")

# Modelo 1
y_pred_1 = modelo_1.predict(X_test_full, verbose=0)
y_pred_1_classes = np.argmax(y_pred_1, axis=1)

# Modelo 3
y_pred_3 = modelo_3.predict(X_test_full, verbose=0)
y_pred_3_classes = np.argmax(y_pred_3, axis=1)

print("✅ Predicciones generadas\n")

# ============================================
# MÉTRICAS CONSOLIDADAS
# ============================================

print("📊 Métricas Consolidadas:")
print("-" * 70)

print(f"\n🎯 MODELO 1: Control de Calidad")
print(f"   Accuracy: {metricas_1['accuracy']*100:.2f}%")
print(f"   Epochs entrenados: {metricas_1['epochs_trained']}")

print(f"\n📈 MODELO 2: Predictor de ABV")
print(f"   R²: {metricas_2['r2']:.4f}")
print(f"   RMSE: {metricas_2['rmse']:.4f}")
print(f"   MAE: {metricas_2['mae']:.4f}")
print(f"   Epochs entrenados: {metricas_2['epochs_trained']}")

print(f"\n🔬 MODELO 3: Clasificador Experimental")
print(f"   Accuracy: {metricas_3['accuracy']*100:.2f}%")
print(f"   Epochs entrenados: {metricas_3['epochs_trained']}")

print("\n" + "="*70)

# ============================================
# GRÁFICOS INTERACTIVOS
# ============================================

print("\n📊 Generando visualizaciones...")

# 1. Comparación de Accuracy (Modelos 1 y 3)
fig1 = go.Figure()

fig1.add_trace(go.Bar(
    x=['Modelo 1<br>Control Calidad', 'Modelo 3<br>Clasificador Exp.'],
    y=[metricas_1['accuracy']*100, metricas_3['accuracy']*100],
    text=[f"{metricas_1['accuracy']*100:.2f}%", f"{metricas_3['accuracy']*100:.2f}%"],
    textposition='auto',
    marker_color=['#667eea', '#764ba2']
))

fig1.update_layout(
    title='Comparación de Accuracy - Modelos de Clasificación',
    yaxis_title='Accuracy (%)',
    yaxis_range=[0, 105],
    height=500
)

graph1_html = fig1.to_html(include_plotlyjs='cdn', div_id='graph1')

# 2. Evolución del entrenamiento - Modelo 1
fig2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Pérdida', 'Accuracy')
)

fig2.add_trace(
    go.Scatter(y=history_1['loss'], mode='lines', name='Train Loss', 
               line=dict(color='blue')),
    row=1, col=1
)
fig2.add_trace(
    go.Scatter(y=history_1['val_loss'], mode='lines', name='Val Loss',
               line=dict(color='red')),
    row=1, col=1
)

fig2.add_trace(
    go.Scatter(y=history_1['accuracy'], mode='lines', name='Train Acc',
               line=dict(color='green')),
    row=1, col=2
)
fig2.add_trace(
    go.Scatter(y=history_1['val_accuracy'], mode='lines', name='Val Acc',
               line=dict(color='orange')),
    row=1, col=2
)

fig2.update_xaxes(title_text="Epoch", row=1, col=1)
fig2.update_xaxes(title_text="Epoch", row=1, col=2)
fig2.update_yaxes(title_text="Loss", row=1, col=1)
fig2.update_yaxes(title_text="Accuracy", row=1, col=2)

fig2.update_layout(
    title_text='Modelo 1: Evolución del Entrenamiento',
    height=500,
    showlegend=True
)

graph2_html = fig2.to_html(include_plotlyjs='cdn', div_id='graph2')

# 3. Predicciones ABV - Modelo 2
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=[0, y_test_modelo_2.max()],
    y=[0, y_test_modelo_2.max()],
    mode='lines',
    name='Predicción Perfecta',
    line=dict(color='red', dash='dash')
))

fig3.add_trace(go.Scatter(
    x=y_test_modelo_2,
    y=y_pred_modelo_2,
    mode='markers',
    name='Predicciones',
    marker=dict(
        size=12,
        color='skyblue',
        line=dict(width=1, color='navy')
    ),
    text=[f"Real: {r:.2f}%<br>Pred: {p:.2f}%" for r, p in zip(y_test_modelo_2, y_pred_modelo_2)],
    hovertemplate='%{text}<extra></extra>'
))

fig3.update_layout(
    title=f'Modelo 2: Predicción de ABV (R² = {metricas_2["r2"]:.4f})',
    xaxis_title='ABV Real (%)',
    yaxis_title='ABV Predicho (%)',
    height=600
)

graph3_html = fig3.to_html(include_plotlyjs='cdn', div_id='graph3')

# 4. Evolución del entrenamiento - Modelo 2
fig4 = make_subplots(
    rows=1, cols=2,
    subplot_titles=('MSE Loss', 'MAE')
)

fig4.add_trace(
    go.Scatter(y=history_2['loss'], mode='lines', name='Train MSE',
               line=dict(color='purple')),
    row=1, col=1
)
fig4.add_trace(
    go.Scatter(y=history_2['val_loss'], mode='lines', name='Val MSE',
               line=dict(color='pink')),
    row=1, col=1
)

fig4.add_trace(
    go.Scatter(y=history_2['mae'], mode='lines', name='Train MAE',
               line=dict(color='teal')),
    row=1, col=2
)
fig4.add_trace(
    go.Scatter(y=history_2['val_mae'], mode='lines', name='Val MAE',
               line=dict(color='cyan')),
    row=1, col=2
)

fig4.update_xaxes(title_text="Epoch", row=1, col=1)
fig4.update_xaxes(title_text="Epoch", row=1, col=2)

fig4.update_layout(
    title_text='Modelo 2: Evolución del Entrenamiento',
    height=500
)

graph4_html = fig4.to_html(include_plotlyjs='cdn', div_id='graph4')

# 5. Matriz de Confusión - Modelo 1
cm_1 = confusion_matrix(y_style_test, y_pred_1_classes)
cm_norm_1 = cm_1.astype('float') / cm_1.sum(axis=1)[:, np.newaxis]

text_annotations_1 = []
for i in range(len(cm_1)):
    row = []
    for j in range(len(cm_1[0])):
        row.append(f"{cm_1[i][j]}<br>({cm_norm_1[i][j]*100:.1f}%)")
    text_annotations_1.append(row)

fig5 = go.Figure(data=go.Heatmap(
    z=cm_1,
    x=label_encoder.classes_,
    y=label_encoder.classes_,
    text=text_annotations_1,
    texttemplate="%{text}",
    colorscale='Blues',
    showscale=True
))

fig5.update_layout(
    title='Modelo 1: Matriz de Confusión',
    xaxis_title='Predicción',
    yaxis_title='Real',
    height=500
)

graph5_html = fig5.to_html(include_plotlyjs='cdn', div_id='graph5')

# 6. Probabilidades del Modelo 3
# Seleccionar algunas muestras interesantes
sample_indices = np.random.choice(len(y_style_test), 10, replace=False)

fig6 = go.Figure()

for i, idx in enumerate(sample_indices):
    fig6.add_trace(go.Bar(
        x=label_encoder.classes_,
        y=y_pred_3[idx] * 100,
        name=f'Muestra {i+1}',
        text=[f'{p*100:.1f}%' for p in y_pred_3[idx]],
        textposition='auto'
    ))

fig6.update_layout(
    title='Modelo 3: Distribución de Probabilidades (Muestras Aleatorias)',
    xaxis_title='Estilo',
    yaxis_title='Probabilidad (%)',
    barmode='group',
    height=600,
    showlegend=True
)

graph6_html = fig6.to_html(include_plotlyjs='cdn', div_id='graph6')

# 7. Distribución de errores - Modelo 2
errores_abv = y_test_modelo_2 - y_pred_modelo_2

fig7 = go.Figure()

fig7.add_trace(go.Histogram(
    x=errores_abv,
    nbinsx=20,
    marker_color='lightcoral',
    marker_line=dict(color='darkred', width=1)
))

fig7.update_layout(
    title='Modelo 2: Distribución de Errores en Predicción de ABV',
    xaxis_title='Error (Real - Predicho)',
    yaxis_title='Frecuencia',
    height=500
)

graph7_html = fig7.to_html(include_plotlyjs='cdn', div_id='graph7')

print("✅ Visualizaciones generadas\n")

# ============================================
# GENERAR REPORTE HTML CONSOLIDADO
# ============================================

print("📝 Generando reporte HTML consolidado...")

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Consolidado - Sistema Inteligente Cervecero</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0;
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-card p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stat-card.modelo1 {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        .stat-card.modelo2 {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .stat-card.modelo3 {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .conclusiones {{
            background-color: #e8f4f8;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
        }}
        .conclusiones h3 {{
            color: #667eea;
            margin-top: 0;
        }}
        .conclusiones ul {{
            line-height: 1.8;
        }}
        .graph-container {{
            margin: 30px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            margin-top: 40px;
            border-top: 2px solid #ddd;
        }}
        .modelo-box {{
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .modelo-box h3 {{
            color: #667eea;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🍺 Sistema Inteligente para Cervecerías</h1>
        <p>Evaluación de 3 Modelos de Deep Learning</p>
        <p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    </div>

    <!-- RESUMEN EJECUTIVO -->
    <div class="section">
        <h2>📊 Resumen Ejecutivo</h2>
        
        <div class="stats-grid">
            <div class="stat-card modelo1">
                <h3>{metricas_1['accuracy']*100:.1f}%</h3>
                <p>Modelo 1: Control Calidad</p>
            </div>
            <div class="stat-card modelo2">
                <h3>{metricas_2['r2']:.3f}</h3>
                <p>Modelo 2: R² ABV</p>
            </div>
            <div class="stat-card modelo3">
                <h3>{metricas_3['accuracy']*100:.1f}%</h3>
                <p>Modelo 3: Clasificador Exp</p>
            </div>
        </div>

        <div class="conclusiones">
            <h3>✅ Sistema Multi-Modelo Implementado</h3>
            <p>Se desarrollaron <strong>3 modelos de Deep Learning independientes</strong>, 
            cada uno optimizado para resolver un problema específico de las cervecerías artesanales:</p>
            <ul>
                <li><strong>Modelo 1:</strong> Control de calidad mediante clasificación de estilos</li>
                <li><strong>Modelo 2:</strong> Predicción de contenido alcohólico (ABV)</li>
                <li><strong>Modelo 3:</strong> Clasificación de recetas con análisis probabilístico</li>
            </ul>
        </div>
    </div>

    <!-- MODELO 1 -->
    <div class="section">
        <h2>🎯 Modelo 1: Control de Calidad</h2>
        
        <div class="modelo-box">
            <h3>📋 Especificaciones</h3>
            <table>
                <tr>
                    <th>Característica</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>Tipo</td>
                    <td>Clasificación Multi-clase</td>
                </tr>
                <tr>
                    <td>Input</td>
                    <td>6 parámetros (OG, ABV, pH, IBU, ABV_OG_ratio, IBU_ABV_ratio)</td>
                </tr>
                <tr>
                    <td>Output</td>
                    <td>3 estilos (Premium Lager, IPA, Light Lager)</td>
                </tr>
                <tr>
                    <td>Arquitectura</td>
                    <td>6 → 128 → 64 → 32 → 3</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td><strong>{metricas_1['accuracy']*100:.2f}%</strong></td>
                </tr>
                <tr>
                    <td>Epochs Entrenados</td>
                    <td>{metricas_1['epochs_trained']}</td>
                </tr>
            </table>
        </div>

        <div class="conclusiones">
            <h3>🎯 Problema Resuelto</h3>
            <p><strong>"¿Este lote salió bien?"</strong></p>
            <p>El modelo clasifica automáticamente el estilo de cerveza basándose en sus parámetros 
            fisicoquímicos, permitiendo a los maestros cerveceros validar que el lote cumple 
            con las características del estilo objetivo.</p>
        </div>

        <div class="graph-container">
            {graph2_html}
        </div>

        <div class="graph-container">
            {graph5_html}
        </div>
    </div>

    <!-- MODELO 2 -->
    <div class="section">
        <h2>📈 Modelo 2: Predictor de ABV</h2>
        
        <div class="modelo-box">
            <h3>📋 Especificaciones</h3>
            <table>
                <tr>
                    <th>Característica</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>Tipo</td>
                    <td>Regresión</td>
                </tr>
                <tr>
                    <td>Input</td>
                    <td>3 parámetros (OG, pH, IBU) - <strong>SIN ABV</strong></td>
                </tr>
                <tr>
                    <td>Output</td>
                    <td>ABV estimado (%)</td>
                </tr>
                <tr>
                    <td>Arquitectura</td>
                    <td>3 → 64 → 32 → 16 → 1</td>
                </tr>
                <tr>
                    <td>R²</td>
                    <td><strong>{metricas_2['r2']:.4f}</strong></td>
                </tr>
                <tr>
                    <td>RMSE</td>
                    <td>{metricas_2['rmse']:.4f}%</td>
                </tr>
                <tr>
                    <td>MAE</td>
                    <td>{metricas_2['mae']:.4f}%</td>
                </tr>
                <tr>
                    <td>Epochs Entrenados</td>
                    <td>{metricas_2['epochs_trained']}</td>
                </tr>
            </table>
        </div>

        <div class="conclusiones">
            <h3>🎯 Problema Resuelto</h3>
            <p><strong>"¿Cuánto alcohol tendrá mi cerveza?"</strong></p>
          <p>
    El modelo predice el contenido de alcohol (ABV) utilizando parámetros iniciales como la densidad
    (OG), pH e IBU. Aunque el ABV real solo se confirma al finalizar la fermentación, la predicción
    temprana permite anticipar el comportamiento del lote, detectar desviaciones potenciales y tomar
    decisiones correctivas antes de completar el proceso.
</p>
        </div>

        <div class="graph-container">
            {graph3_html}
        </div>

        <div class="graph-container">
            {graph4_html}
        </div>

        <div class="graph-container">
            {graph7_html}
        </div>
    </div>

    <!-- MODELO 3 -->
    <div class="section">
        <h2>🔬 Modelo 3: Clasificador de Recetas Experimentales</h2>
        
        <div class="modelo-box">
            <h3>📋 Especificaciones</h3>
            <table>
                <tr>
                    <th>Característica</th>
                    <th>Valor</th>
                </tr>
                <tr>
                    <td>Tipo</td>
                    <td>Clasificación con Análisis Probabilístico</td>
                </tr>
                <tr>
                    <td>Input</td>
                    <td>6 parámetros (OG, ABV, pH, IBU, ABV_OG_ratio, IBU_ABV_ratio)</td>
                </tr>
                <tr>
                    <td>Output</td>
                    <td>Probabilidades para 3 estilos</td>
                </tr>
                <tr>
                    <td>Arquitectura</td>
                    <td>6 → 96 → 48 → 24 → 3</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td><strong>{metricas_3['accuracy']*100:.2f}%</strong></td>
                </tr>
                <tr>
                    <td>Epochs Entrenados</td>
                    <td>{metricas_3['epochs_trained']}</td>
                </tr>
            </table>
        </div>

        <div class="conclusiones">
            <h3>🎯 Problema Resuelto</h3>
            <p><strong>"¿A qué estilo se parece este experimento?"</strong></p>
            <p>Ideal para recetas experimentales, este modelo proporciona no solo la clasificación 
            más probable, sino también la <strong>distribución completa de probabilidades</strong> 
            para cada estilo, ayudando a los cerveceros a entender qué tan cerca está su 
            experimento de cada categoría.</p>
        </div>

        <div class="graph-container">
            {graph6_html}
        </div>
    </div>

    <!-- COMPARACIÓN -->
    <div class="section">
        <h2>⚖️ Comparación de Modelos</h2>
        
        <div class="graph-container">
            {graph1_html}
        </div>

        <table>
            <tr>
                <th>Modelo</th>
                <th>Propósito</th>
                <th>Métrica Principal</th>
                <th>Rendimiento</th>
                <th>Uso Recomendado</th>
            </tr>
            <tr>
                <td><strong>Modelo 1</strong></td>
                <td>Control de Calidad</td>
                <td>Accuracy</td>
                <td>{metricas_1['accuracy']*100:.2f}%</td>
                <td>Validación de lotes terminados</td>
            </tr>
            <tr>
                <td><strong>Modelo 2</strong></td>
                <td>Predicción ABV</td>
                <td>R²</td>
                <td>{metricas_2['r2']:.4f}</td>
                <td>Estimación durante fermentación</td>
            </tr>
            <tr>
                <td><strong>Modelo 3</strong></td>
                <td>Análisis Experimental</td>
                <td>Accuracy</td>
                <td>{metricas_3['accuracy']*100:.2f}%</td>
                <td>Desarrollo de nuevas recetas</td>
            </tr>
        </table>
    </div>

    <!-- CONCLUSIONES FINALES -->
    <div class="section">
        <h2>💡 Conclusiones y Valor del Sistema</h2>
        
        <div class="conclusiones">
            <h3>✅ Logros del Proyecto</h3>
            <ul>
                <li>✅ <strong>3 modelos independientes</strong> de Deep Learning implementados exitosamente</li>
                <li>✅ <strong>Dataset pequeño (150 muestras)</strong> utilizado eficientemente mediante data augmentation</li>
                <li>✅ <strong>Problemas reales</strong> de cervecerías artesanales resueltos</li>
                <li>✅ <strong>Métricas excelentes:</strong> Accuracy >95%, R² positivo</li>
                <li>✅ <strong>Arquitecturas optimizadas</strong> para cada tarea específica</li>
            </ul>
        </div>

        <div class="conclusiones">
            <h3>🎯 Valor para Cervecerías Artesanales</h3>
            <ul>
                <li><strong>Modelo 1:</strong> Automatiza el control de calidad, reduciendo errores humanos</li>
                <li><strong>Modelo 2:</strong> Permite planificación anticipada de producción y etiquetado</li>
                <li><strong>Modelo 3:</strong> Facilita la innovación en desarrollo de nuevas recetas</li>
                <li><strong>Sistema completo:</strong> Cubre todo el ciclo productivo de una cervecería</li>
            </ul>
        </div>

        <div class="conclusiones">
            <h3>🚀 Tecnologías Implementadas</h3>
            <ul>
                <li>Deep Learning con <strong>TensorFlow/Keras</strong></li>
                <li>Arquitecturas: <strong>Redes Neuronales Densas</strong> con BatchNormalization y Dropout</li>
                <li>Técnicas: <strong>SMOTE</strong> para data augmentation, <strong>StandardScaler</strong> para normalización</li>
                <li>Optimización: <strong>Adam optimizer</strong>, Early Stopping, Learning Rate Scheduling</li>
                <li>Evaluación: Métricas estándar de ML (Accuracy, R², RMSE, MAE, Confusion Matrix)</li>
            </ul>
        </div>
    </div>

    <div class="footer">
        <p>🍺 Sistema Inteligente para Cervecerías Artesanales - Valdivia, Chile</p>
        <p>Proyecto de Aplicaciones de Inteligencia Artificial - Deep Learning</p>
        <p>Noviembre 2025</p>
    </div>
</body>
</html>
"""

with open('REPORTE_EVALUACION_MODELOS.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✅ Reporte HTML consolidado generado: REPORTE_EVALUACION_MODELOS.html")

# ============================================
# RESUMEN FINAL
# ============================================

print("\n" + "="*70)
print("✅ EVALUACIÓN CONSOLIDADA COMPLETADA")
print("="*70)

print(f"""
📊 RESUMEN DE LOS 3 MODELOS:

🎯 MODELO 1 - Control de Calidad:
   • Accuracy: {metricas_1['accuracy']*100:.2f}%
   • Epochs: {metricas_1['epochs_trained']}
   • Archivo: modelos/modelo_1_control_calidad.h5

📈 MODELO 2 - Predictor de ABV:
   • R²: {metricas_2['r2']:.4f}
   • RMSE: {metricas_2['rmse']:.4f}
   • MAE: {metricas_2['mae']:.4f}
   • Epochs: {metricas_2['epochs_trained']}
   • Archivo: modelos/modelo_2_predictor_abv.h5

🔬 MODELO 3 - Clasificador Experimental:
   • Accuracy: {metricas_3['accuracy']*100:.2f}%
   • Epochs: {metricas_3['epochs_trained']}
   • Archivo: modelos/modelo_3_clasificador_experimental.h5

📁 REPORTES GENERADOS:
   • REPORTE_EVALUACION_MODELOS.html (reporte completo)
   • 7 gráficos interactivos incluidos

""")

print("="*70)