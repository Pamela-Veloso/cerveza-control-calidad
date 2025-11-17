"""
MODELO 2: PREDICTOR DE ABV
Regresión para predecir contenido de alcohol
Input: OG, pH, IBU (SIN ABV)
Output: ABV estimado
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

print("="*70)
print("🍺 MODELO 2: PREDICTOR DE ABV")
print("="*70)

# Cargar datos
X_train_full = np.load('data/X_train.npy')
X_val_full = np.load('data/X_val.npy')
X_test_full = np.load('data/X_test.npy')

# Para este modelo, NO usamos ABV como input (columna 1)
# Columnas: [0:OG, 1:ABV, 2:pH, 3:IBU, 4:ABV_OG_ratio, 5:IBU_ABV_ratio]
# Usamos: OG, pH, IBU (índices 0, 2, 3)
X_train = X_train_full[:, [0, 2, 3]]
X_val = X_val_full[:, [0, 2, 3]]
X_test = X_test_full[:, [0, 2, 3]]

# Target: ABV (extraer del dataset original)
# Necesitamos reconstruir ABV desde los datos normalizados
# Cargar scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Desnormalizar columna ABV (índice 1)
X_train_denorm = scaler.inverse_transform(X_train_full)
X_val_denorm = scaler.inverse_transform(X_val_full)
X_test_denorm = scaler.inverse_transform(X_test_full)

y_train = X_train_denorm[:, 1]  # ABV
y_val = X_val_denorm[:, 1]
y_test = X_test_denorm[:, 1]

print(f"\n✅ Datos preparados:")
print(f"   Input shape: {X_train.shape} (OG, pH, IBU)")
print(f"   Output: ABV [{y_train.min():.2f}, {y_train.max():.2f}]")

# Crear modelo
def crear_modelo_predictor_abv():
    model = models.Sequential([
        layers.Input(shape=(3,)),  # Solo 3 inputs
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='linear')  # Regresión
    ], name='Predictor_ABV_Model')
    
    return model

modelo = crear_modelo_predictor_abv()
modelo.summary()

# Compilar
modelo.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae']
)

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20),
    callbacks.ModelCheckpoint('modelos/modelo_2_predictor_abv.h5', save_best_only=True)
]

# Entrenar
print("\n🚀 Entrenando Modelo 2...")
history = modelo.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluar
y_pred = modelo.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*70)
print("📊 RESULTADOS MODELO 2")
print("="*70)
print(f"\n✅ R²: {r2:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ MAE: {mae:.4f}")

print("\n🔍 Ejemplos de predicciones:")
for i in range(min(10, len(y_test))):
    print(f"   Real: {y_test[i]:.2f}% → Predicho: {y_pred[i]:.2f}% (Error: {abs(y_test[i]-y_pred[i]):.2f}%)")

# Guardar métricas
metricas = {
    'r2': float(r2),
    'rmse': float(rmse),
    'mae': float(mae),
    'mse': float(mse),
    'epochs_trained': len(history.history['loss'])
}

with open('modelos/modelo_2_metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)

with open('modelos/modelo_2_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

# Guardar predicciones
np.save('modelos/modelo_2_y_test.npy', y_test)
np.save('modelos/modelo_2_y_pred.npy', y_pred)

print("\n💾 Modelo 2 guardado: modelos/modelo_2_predictor_abv.h5")
print("="*70)