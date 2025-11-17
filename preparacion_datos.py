"""
PREPARACIÓN DE DATOS PARA EL MODELO
- Feature Engineering
- Normalización
- Data Augmentation (SMOTE 2x)
- Train/Val/Test Split
- Generación de Score de Calidad
- REPORTE HTML
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Crear carpetas necesarias
for folder in ['visualizaciones', 'data']:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✅ Carpeta '{folder}/' creada")

print("="*70)
print("🔧 PREPARACIÓN DE DATOS PARA DEEP LEARNING")
print("="*70)

# ============================================
# PASO 1: CARGAR DATOS ORIGINALES
# ============================================

print("\n📁 Cargando dataset original...")
df = pd.read_csv('beer.csv')
print(f"✅ Dataset cargado: {df.shape[0]} registros")

# Mostrar distribución original
print("\n📊 Distribución original por estilo:")
dist_original = df['style'].value_counts()
print(dist_original)

# ============================================
# PASO 2: FEATURE ENGINEERING
# ============================================

print("\n" + "="*70)
print("🛠️ FEATURE ENGINEERING - CREANDO NUEVAS VARIABLES")
print("="*70)

# Variables originales
X_original = df[['OG', 'ABV', 'pH', 'IBU']].values
y_style = df['style'].values

print("\n✅ Variables originales: OG, ABV, pH, IBU")

# Crear nuevas características (ratios)
df['ABV_OG_ratio'] = df['ABV'] / df['OG']  # Eficiencia de fermentación
df['IBU_ABV_ratio'] = df['IBU'] / df['ABV']  # Balance amargor/alcohol

print("✅ Nuevas características creadas:")
print("   • ABV_OG_ratio: Eficiencia de fermentación")
print("   • IBU_ABV_ratio: Balance amargor/alcohol")

# Estadísticas de nuevas features
new_features_stats = df[['ABV_OG_ratio', 'IBU_ABV_ratio']].describe()

# Matriz de características AMPLIADA
X_features = df[['OG', 'ABV', 'pH', 'IBU', 'ABV_OG_ratio', 'IBU_ABV_ratio']].values

print(f"\n📊 Matriz de características: {X_features.shape}")
print(f"   • Variables originales: 4")
print(f"   • Variables nuevas: 2")
print(f"   • TOTAL: 6 características")

# ============================================
# PASO 3: CODIFICAR ETIQUETAS
# ============================================

print("\n" + "="*70)
print("🔢 CODIFICACIÓN DE ETIQUETAS")
print("="*70)

# Codificar estilos a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_style)

print("\n✅ Mapeo de estilos:")
mapeo_estilos = {}
for idx, style in enumerate(label_encoder.classes_):
    print(f"   {style} → {idx}")
    mapeo_estilos[style] = idx

# Guardar el encoder para usar después
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("\n💾 Label encoder guardado en: label_encoder.pkl")

# ============================================
# PASO 4: GENERAR SCORE DE CALIDAD
# ============================================

print("\n" + "="*70)
print("⭐ GENERACIÓN DE SCORE DE CALIDAD (0-100)")
print("="*70)

def calcular_score_calidad(row):
    """
    Calcula un score de calidad basado en qué tan cerca están
    los parámetros de los rangos óptimos para cada estilo
    """
    style = row['style']
    
    # Rangos óptimos definidos por maestros cerveceros
    # (basados en el análisis exploratorio)
    optimal_ranges = {
        'Premium Lager': {
            'OG': (11.4, 12.6),
            'ABV': (3.9, 4.8),
            'pH': (3.9, 4.25),
            'IBU': (12, 14)
        },
        'IPA': {
            'OG': (13.0, 14.5),
            'ABV': (6.3, 6.98),
            'pH': (3.8, 4.5),
            'IBU': (38, 50)
        },
        'Light Lager': {
            'OG': (9.6, 10.2),
            'ABV': (2.7, 3.3),
            'pH': (3.8, 4.25),
            'IBU': (8, 10)
        }
    }
    
    ranges = optimal_ranges[style]
    scores = []
    
    # Calcular score para cada parámetro
    for param in ['OG', 'ABV', 'pH', 'IBU']:
        value = row[param]
        min_val, max_val = ranges[param]
        center = (min_val + max_val) / 2
        range_width = max_val - min_val
        
        # Distancia al centro del rango óptimo
        distance = abs(value - center)
        
        # Score: 100 si está en el centro, decrece con la distancia
        if min_val <= value <= max_val:
            param_score = 100 * (1 - distance / (range_width / 2))
        else:
            excess = min(distance - range_width / 2, range_width)
            param_score = max(0, 70 - (excess / range_width) * 70)
        
        scores.append(param_score)
    
    # Score final: promedio ponderado
    final_score = (
        scores[0] * 0.15 +  # OG
        scores[1] * 0.35 +  # ABV
        scores[2] * 0.15 +  # pH
        scores[3] * 0.35    # IBU
    )
    
    return final_score / 100  # Normalizar a 0-1

# Calcular score para cada registro
df['quality_score'] = df.apply(calcular_score_calidad, axis=1)

print("\n✅ Score de calidad generado para todos los registros")
print(f"\n📊 Estadísticas del Score de Calidad:")
quality_stats = df['quality_score'].describe()
print(quality_stats)

print("\n📈 Distribución de scores por estilo:")
scores_por_estilo = {}
for style in df['style'].unique():
    scores = df[df['style'] == style]['quality_score']
    scores_por_estilo[style] = {
        'mean': scores.mean(),
        'min': scores.min(),
        'max': scores.max(),
        'std': scores.std()
    }
    print(f"   {style}:")
    print(f"      Media: {scores.mean():.3f}")
    print(f"      Min: {scores.min():.3f}, Max: {scores.max():.3f}")

# Gráfico 1: Distribución de scores
fig1 = px.histogram(
    df,
    x='quality_score',
    nbins=30,
    title='Distribución del Score de Calidad',
    labels={'quality_score': 'Score de Calidad'},
    color_discrete_sequence=['skyblue']
)
graph1_html = fig1.to_html(include_plotlyjs='cdn', div_id='graph1')

# Gráfico 2: Box plot por estilo
fig2 = px.box(
    df,
    x='style',
    y='quality_score',
    color='style',
    title='Score de Calidad por Estilo',
    labels={'quality_score': 'Score de Calidad', 'style': 'Estilo'},
    points='all'
)
graph2_html = fig2.to_html(include_plotlyjs='cdn', div_id='graph2')

# ============================================
# PASO 5: NORMALIZACIÓN
# ============================================

print("\n" + "="*70)
print("📏 NORMALIZACIÓN DE DATOS")
print("="*70)

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print("\n✅ StandardScaler aplicado")
print(f"   Media de características normalizadas: {X_scaled.mean(axis=0).round(3)}")
print(f"   Desviación estándar: {X_scaled.std(axis=0).round(3)}")

# Guardar el scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("\n💾 Scaler guardado en: scaler.pkl")

# ============================================
# PASO 6: DIVISIÓN TRAIN/VAL/TEST
# ============================================

print("\n" + "="*70)
print("✂️ DIVISIÓN DE DATOS")
print("="*70)

# Preparar targets
y_quality = df['quality_score'].values

# Primera división: Train+Val (85%) y Test (15%)
X_temp, X_test, y_style_temp, y_style_test, y_quality_temp, y_quality_test = train_test_split(
    X_scaled, y_encoded, y_quality,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)

# Segunda división: Train (70% del total) y Val (15% del total)
X_train, X_val, y_style_train, y_style_val, y_quality_train, y_quality_val = train_test_split(
    X_temp, y_style_temp, y_quality_temp,
    test_size=0.176,  # 15% del total = 0.176 de 85%
    random_state=42,
    stratify=y_style_temp
)

train_size = X_train.shape[0]
val_size = X_val.shape[0]
test_size = X_test.shape[0]

print(f"\n✅ División completada:")
print(f"   📦 Train: {train_size} muestras ({train_size/len(df)*100:.1f}%)")
print(f"   📦 Validation: {val_size} muestras ({val_size/len(df)*100:.1f}%)")
print(f"   📦 Test: {test_size} muestras ({test_size/len(df)*100:.1f}%)")

# Verificar balanceo
print("\n📊 Distribución de estilos en cada conjunto:")
distribucion_splits = {}
for name, y_data in [('Train', y_style_train), ('Val', y_style_val), ('Test', y_style_test)]:
    unique, counts = np.unique(y_data, return_counts=True)
    distribucion_splits[name] = {}
    print(f"\n   {name}:")
    for style_idx, count in zip(unique, counts):
        style_name = label_encoder.classes_[style_idx]
        distribucion_splits[name][style_name] = count
        print(f"      {style_name}: {count} ({count/len(y_data)*100:.1f}%)")

# Gráfico 3: Distribución de splits
split_data = []
for split_name, styles in distribucion_splits.items():
    for style, count in styles.items():
        split_data.append({'Split': split_name, 'Estilo': style, 'Cantidad': count})

df_splits = pd.DataFrame(split_data)
fig3 = px.bar(
    df_splits,
    x='Split',
    y='Cantidad',
    color='Estilo',
    title='Distribución de Estilos por Conjunto (Antes de SMOTE)',
    barmode='group'
)
graph3_html = fig3.to_html(include_plotlyjs='cdn', div_id='graph3')

# ============================================
# PASO 7: DATA AUGMENTATION (SMOTE 2x)
# ============================================

print("\n" + "="*70)
print("🔄 DATA AUGMENTATION CON SMOTE (2x)")
print("="*70)

train_original_size = X_train.shape[0]
print(f"\n📊 Antes de SMOTE: {train_original_size} muestras")

# Contar muestras actuales por clase
unique_train, counts_train = np.unique(y_style_train, return_counts=True)
print("\n📊 Distribución original en Train:")
for style_idx, count in zip(unique_train, counts_train):
    style_name = label_encoder.classes_[style_idx]
    print(f"   {style_name}: {count} muestras")

# Calcular cuántas muestras queremos por clase (DUPLICAR = 2x)
samples_per_class = int(train_original_size / len(unique_train) * 2)

# Crear diccionario de sampling strategy
sampling_strategy = {i: samples_per_class for i in unique_train}

print(f"\n🎯 Objetivo SMOTE 2x: {samples_per_class} muestras por clase")
print(f"   Total esperado: {samples_per_class * len(unique_train)} muestras")

# Aplicar SMOTE con estrategia personalizada
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=5,
    random_state=42
)

X_train_augmented, y_style_train_augmented = smote.fit_resample(
    X_train, 
    y_style_train
)

# Para el quality score, interpolamos
print("\n🔄 Generando quality scores para muestras sintéticas...")
quality_scores_augmented = []
for i in range(len(y_style_train_augmented)):
    style = y_style_train_augmented[i]
    similar_indices = np.where(y_style_train == style)[0]
    
    if i < len(y_style_train):
        # Muestra original
        quality_scores_augmented.append(y_quality_train[i])
    else:
        # Muestra sintética: promedio de scores similares con ruido
        similar_scores = y_quality_train[similar_indices]
        base_score = np.mean(similar_scores)
        noise = np.random.normal(0, 0.05)  # Pequeña variación
        quality_scores_augmented.append(np.clip(base_score + noise, 0, 1))

y_quality_train_augmented = np.array(quality_scores_augmented)

train_augmented_size = X_train_augmented.shape[0]
incremento = train_augmented_size - train_original_size

print(f"\n✅ Después de SMOTE 2x: {train_augmented_size} muestras")
print(f"   Incremento: +{incremento} muestras ({incremento/train_original_size*100:.1f}%)")
print(f"   Factor de multiplicación: {train_augmented_size/train_original_size:.2f}x")

# Verificar nuevo balanceo
print("\n📊 Distribución después de SMOTE:")
distribucion_smote = {}
unique, counts = np.unique(y_style_train_augmented, return_counts=True)
for style_idx, count in zip(unique, counts):
    style_name = label_encoder.classes_[style_idx]
    distribucion_smote[style_name] = count
    print(f"   {style_name}: {count} ({count/len(y_style_train_augmented)*100:.1f}%)")

# Gráfico 4: Comparación antes/después SMOTE
smote_comparison_data = []
for style in label_encoder.classes_:
    style_idx = mapeo_estilos[style]
    original = np.sum(y_style_train == style_idx)
    augmented = np.sum(y_style_train_augmented == style_idx)
    smote_comparison_data.append({'Estilo': style, 'Momento': 'Antes SMOTE', 'Cantidad': original})
    smote_comparison_data.append({'Estilo': style, 'Momento': 'Después SMOTE 2x', 'Cantidad': augmented})

df_smote = pd.DataFrame(smote_comparison_data)
fig4 = px.bar(
    df_smote,
    x='Estilo',
    y='Cantidad',
    color='Momento',
    title='Efecto de SMOTE 2x en el Conjunto de Entrenamiento',
    barmode='group',
    color_discrete_sequence=['#ff7f0e', '#2ca02c']
)
graph4_html = fig4.to_html(include_plotlyjs='cdn', div_id='graph4')

# ============================================
# PASO 8: GUARDAR DATOS PROCESADOS
# ============================================

print("\n" + "="*70)
print("💾 GUARDANDO DATOS PROCESADOS")
print("="*70)

# Guardar todos los conjuntos
np.save('data/X_train.npy', X_train_augmented)
np.save('data/X_val.npy', X_val)
np.save('data/X_test.npy', X_test)

np.save('data/y_style_train.npy', y_style_train_augmented)
np.save('data/y_style_val.npy', y_style_val)
np.save('data/y_style_test.npy', y_style_test)

np.save('data/y_quality_train.npy', y_quality_train_augmented)
np.save('data/y_quality_val.npy', y_quality_val)
np.save('data/y_quality_test.npy', y_quality_test)

print("\n✅ Datos guardados en carpeta 'data/':")
print("   • X_train.npy, X_val.npy, X_test.npy")
print("   • y_style_train.npy, y_style_val.npy, y_style_test.npy")
print("   • y_quality_train.npy, y_quality_val.npy, y_quality_test.npy")

# ============================================
# PASO 9: GENERAR REPORTE HTML
# ============================================

print("\n" + "="*70)
print("📝 GENERANDO REPORTE HTML")
print("="*70)

html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte Preparación de Datos</title>
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
        .highlight {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 Preparación de Datos</h1>
        <p>Sistema de Control de Calidad Cervecero</p>
        <p>Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
    </div>

    <!-- RESUMEN EJECUTIVO -->
    <div class="section">
        <h2>📊 Resumen Ejecutivo</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>6</h3>
                <p>Features Totales</p>
            </div>
            <div class="stat-card">
                <h3>{train_augmented_size}</h3>
                <p>Muestras de Entrenamiento</p>
            </div>
            <div class="stat-card">
                <h3>{val_size}</h3>
                <p>Muestras de Validación</p>
            </div>
            <div class="stat-card">
                <h3>{test_size}</h3>
                <p>Muestras de Prueba</p>
            </div>
        </div>

        <div class="highlight">
            <strong>🎯 Data Augmentation Aplicado:</strong> SMOTE 2x duplicó el dataset de entrenamiento
            de {train_original_size} a {train_augmented_size} muestras (+{incremento} sintéticas).
            Esto mejorará significativamente la capacidad de generalización del modelo.
        </div>
    </div>

    <!-- FEATURE ENGINEERING -->
    <div class="section">
        <h2>🛠️ Feature Engineering</h2>
        
        <div class="conclusiones">
            <h3>✅ Variables Originales (4):</h3>
            <ul>
                <li><strong>OG</strong> (Original Gravity): Densidad inicial del mosto</li>
                <li><strong>ABV</strong> (Alcohol By Volume): Porcentaje de alcohol</li>
                <li><strong>pH</strong>: Nivel de acidez</li>
                <li><strong>IBU</strong> (International Bitterness Units): Amargor</li>
            </ul>
        </div>

        <div class="conclusiones">
            <h3>🆕 Variables Creadas (2):</h3>
            <ul>
                <li><strong>ABV_OG_ratio</strong>: Eficiencia de fermentación (ABV/OG)</li>
                <li><strong>IBU_ABV_ratio</strong>: Balance entre amargor y alcohol (IBU/ABV)</li>
            </ul>
        </div>

        <table>
            <tr>
                <th>Estadística</th>
                <th>ABV_OG_ratio</th>
                <th>IBU_ABV_ratio</th>
            </tr>
            <tr>
                <td>Media</td>
                <td>{new_features_stats.loc['mean', 'ABV_OG_ratio']:.4f}</td>
                <td>{new_features_stats.loc['mean', 'IBU_ABV_ratio']:.4f}</td>
            </tr>
            <tr>
                <td>Desv. Estándar</td>
                <td>{new_features_stats.loc['std', 'ABV_OG_ratio']:.4f}</td>
                <td>{new_features_stats.loc['std', 'IBU_ABV_ratio']:.4f}</td>
            </tr>
            <tr>
                <td>Mínimo</td>
                <td>{new_features_stats.loc['min', 'ABV_OG_ratio']:.4f}</td>
                <td>{new_features_stats.loc['min', 'IBU_ABV_ratio']:.4f}</td>
            </tr>
            <tr>
                <td>Máximo</td>
                <td>{new_features_stats.loc['max', 'ABV_OG_ratio']:.4f}</td>
                <td>{new_features_stats.loc['max', 'IBU_ABV_ratio']:.4f}</td>
            </tr>
        </table>
    </div>

    <!-- SCORE DE CALIDAD -->
    <div class="section">
        <h2>⭐ Score de Calidad</h2>
        
        <div class="conclusiones">
            <h3>💡 ¿Cómo se calcula?</h3>
            <p>El score de calidad (0-1) se calcula midiendo qué tan cerca están los parámetros 
            de los rangos óptimos para cada estilo, con los siguientes pesos:</p>
            <ul>
                <li><strong>OG:</strong> 15% del score</li>
                <li><strong>ABV:</strong> 35% del score (CRÍTICO)</li>
                <li><strong>pH:</strong> 15% del score</li>
                <li><strong>IBU:</strong> 35% del score (CRÍTICO)</li>
            </ul>
        </div>

        <table>
            <tr>
                <th>Estilo</th>
                <th>Score Promedio</th>
                <th>Score Mínimo</th>
                <th>Score Máximo</th>
                <th>Desv. Estándar</th>
            </tr>
            {''.join([f'''<tr>
                <td>{style}</td>
                <td>{scores_por_estilo[style]["mean"]:.3f}</td>
                <td>{scores_por_estilo[style]["min"]:.3f}</td>
                <td>{scores_por_estilo[style]["max"]:.3f}</td>
                <td>{scores_por_estilo[style]["std"]:.3f}</td>
            </tr>''' for style in label_encoder.classes_])}
        </table>

        <div class="graph-container">
            {graph1_html}
        </div>

        <div class="graph-container">
            {graph2_html}
        </div>
    </div>

    <!-- NORMALIZACIÓN -->
    <div class="section">
        <h2>📏 Normalización de Datos</h2>
        
        <div class="conclusiones">
            <h3>✅ StandardScaler Aplicado</h3>
            <p>Se aplicó StandardScaler para normalizar todas las características a media 0 y desviación estándar 1.</p>
            <p>Esto es crucial para redes neuronales porque:</p>
            <ul>
                <li>Acelera la convergencia del entrenamiento</li>
                <li>Evita que variables con rangos grandes dominen el aprendizaje</li>
                <li>Mejora la estabilidad numérica</li>
                <li>Permite que todos los features contribuyan equitativamente</li>
            </ul>
        </div>
    </div>

    <!-- DIVISIÓN DE DATOS -->
    <div class="section">
        <h2>✂️ División de Datos</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{train_original_size}</h3>
                <p>Train Original (70%)</p>
            </div>
            <div class="stat-card">
                <h3>{val_size}</h3>
                <p>Validation (15%)</p>
            </div>
            <div class="stat-card">
                <h3>{test_size}</h3>
                <p>Test (15%)</p>
            </div>
        </div>

        <div class="graph-container">
            {graph3_html}
        </div>

        <div class="conclusiones">
            <h3>📊 Estrategia de División:</h3>
            <ul>
                <li><strong>Train:</strong> Para entrenar el modelo (se le aplicará SMOTE)</li>
                <li><strong>Validation:</strong> Para ajustar hiperparámetros y evitar overfitting</li>
                <li><strong>Test:</strong> Para evaluación final (nunca visto por el modelo)</li>
                <li><strong>División estratificada:</strong> Mantiene proporción de estilos en cada conjunto</li>
            </ul>
        </div>
    </div>

    <!-- DATA AUGMENTATION -->
    <div class="section">
        <h2>🔄 Data Augmentation con SMOTE 2x</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>{train_original_size}</h3>
                <p>Antes de SMOTE</p>
            </div>
            <div class="stat-card">
                <h3>{train_augmented_size}</h3>
                <p>Después de SMOTE</p>
            </div>
            <div class="stat-card">
                <h3>+{incremento}</h3>
                <p>Muestras Generadas</p>
            </div>
            <div class="stat-card">
                <h3>{train_augmented_size/train_original_size:.2f}x</h3>
                <p>Factor Multiplicador</p>
            </div>
        </div>

        <div class="graph-container">
            {graph4_html}
        </div>

        <div class="conclusiones">
            <h3>💡 ¿Qué es SMOTE?</h3>
            <p><strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> genera nuevas muestras sintéticas 
            interpolando entre muestras existentes del mismo estilo.</p>
            <p><strong>Beneficios de SMOTE 2x:</strong></p>
            <ul>
                <li>✅ <strong>Duplica el dataset de entrenamiento</strong> de {train_original_size} a {train_augmented_size} muestras</li>
                <li>✅ <strong>Reduce overfitting:</strong> Más datos = mejor generalización</li>
                <li>✅ <strong>Mantiene el balance:</strong> Cada estilo tiene el mismo número de muestras</li>
                <li>✅ <strong>Mejora la robustez:</strong> El modelo aprende de más variaciones</li>
                <li>✅ <strong>Estándar de la industria:</strong> Práctica común con datasets pequeños</li>
            </ul>
        </div>

        <div class="highlight">
            <strong>⚠️ Importante:</strong> SMOTE solo se aplica al conjunto de ENTRENAMIENTO. 
            Los conjuntos de validación y test se mantienen con datos 100% originales para 
            evaluar correctamente la capacidad de generalización del modelo.
        </div>
    </div>

    <!-- ARCHIVOS GENERADOS -->
    <div class="section">
        <h2>💾 Archivos Generados</h2>
        
        <table>
            <tr>
                <th>Archivo</th>
                <th>Descripción</th>
                <th>Dimensiones</th>
            </tr>
            <tr>
                <td><strong>scaler.pkl</strong></td>
                <td>StandardScaler entrenado</td>
                <td>Para normalizar nuevos datos</td>
            </tr>
            <tr>
                <td><strong>label_encoder.pkl</strong></td>
                <td>Codificador de estilos</td>
                <td>Mapeo: estilo ↔ número</td>
            </tr>
            <tr>
                <td><strong>data/X_train.npy</strong></td>
                <td>Características de entrenamiento (CON SMOTE)</td>
                <td>{train_augmented_size} × 6</td>
            </tr>
            <tr>
                <td><strong>data/X_val.npy</strong></td>
                <td>Características de validación (ORIGINAL)</td>
                <td>{val_size} × 6</td>
            </tr>
            <tr>
                <td><strong>data/X_test.npy</strong></td>
                <td>Características de prueba (ORIGINAL)</td>
                <td>{test_size} × 6</td>
            </tr>
            <tr>
                <td><strong>data/y_style_*.npy</strong></td>
                <td>Etiquetas de estilos (3 archivos)</td>
                <td>Para clasificación</td>
            </tr>
            <tr>
                <td><strong>data/y_quality_*.npy</strong></td>
                <td>Scores de calidad (3 archivos)</td>
                <td>Para regresión</td>
            </tr>
        </table>
    </div>

    <!-- CONCLUSIONES -->
    <div class="section">
        <h2>🎯 Conclusiones</h2>
        
        <div class="conclusiones">
            <h3>✅ Preparación Exitosa</h3>
            <ul>
                <li>Dataset ampliado de {train_original_size} a {train_augmented_size} muestras de entrenamiento</li>
                <li>Total de {train_augmented_size + val_size + test_size} muestras en todos los conjuntos</li>
                <li>6 características por muestra (4 originales + 2 creadas)</li>
                <li>2 objetivos: clasificación (estilo) + regresión (score calidad)</li>
                <li>Datos normalizados y perfectamente balanceados</li>
                <li>División estratificada manteniendo proporciones</li>
            </ul>
        </div>

        <div class="conclusiones">
            <h3>🚀 Ventajas de esta Preparación</h3>
            <ul>
                <li><strong>Más datos de entrenamiento:</strong> {train_augmented_size/train_original_size:.1f}x más muestras para aprender patrones</li>
                <li><strong>Mejor generalización:</strong> Reduce riesgo de overfitting</li>
                <li><strong>Resultados más estables:</strong> Menor varianza entre entrenamientos</li>
                <li><strong>Académicamente sólido:</strong> SMOTE es práctica estándar reconocida</li>
            </ul>
        </div>

        </div>

    <div class="footer">
        <p>🍺 Sistema de Control de Calidad Cervecero - Valdivia, Chile</p>
        <p>Fase 2: Preparación de Datos con SMOTE 2x</p>
        <p>Noviembre 2025</p>
    </div>
</body>
</html>
"""

with open('REPORTE_PREPARACION_DATOS.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\n✅ Reporte HTML generado: REPORTE_PREPARACION_DATOS.html")

# ============================================
# PASO 10: RESUMEN FINAL
# ============================================

print("\n" + "="*70)
print("📋 RESUMEN DE LA PREPARACIÓN DE DATOS")
print("="*70)

print(f"""
✅ PROCESAMIENTO COMPLETADO CON SMOTE 2x

📊 Características:
   • Variables originales: 4 (OG, ABV, pH, IBU)
   • Variables creadas: 2 (ABV_OG_ratio, IBU_ABV_ratio)
   • TOTAL: 6 características

🎯 Targets:
   • Clasificación: 3 estilos (Premium Lager, IPA, Light Lager)
   • Regresión: Score de calidad (0-1)

📦 División de datos:
   • Train ORIGINAL: {train_original_size} muestras
   • Train CON SMOTE 2x: {train_augmented_size} muestras (+{incremento})
   • Validation: {val_size} muestras (sin augmentation)
   • Test: {test_size} muestras (sin augmentation)

🔄 Data Augmentation:
   • Técnica: SMOTE 2x
   • Muestras añadidas: {incremento} ({incremento/train_original_size*100:.1f}%)
   • Factor: {train_augmented_size/train_original_size:.2f}x

💾 Archivos generados:
   • scaler.pkl (StandardScaler)
   • label_encoder.pkl (codificador de estilos)
   • Carpeta data/ con todos los conjuntos (9 archivos .npy)
   • REPORTE_PREPARACION_DATOS.html

""")

print("="*70)