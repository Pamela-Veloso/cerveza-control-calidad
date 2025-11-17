"""
MODELO 1: CONTROL DE CALIDAD
Clasificación de estilo de cerveza
Input: OG, ABV, pH, IBU
Output: Estilo (Premium Lager / IPA / Light Lager)
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

print("="*70)
print("🍺 MODELO 1: CONTROL DE CALIDAD")
print("="*70)

# Cargar datos
X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')
X_test = np.load('data/X_test.npy')

y_style_train = np.load('data/y_style_train.npy')
y_style_val = np.load('data/y_style_val.npy')
y_style_test = np.load('data/y_style_test.npy')

# One-hot encoding
y_train_cat = keras.utils.to_categorical(y_style_train, 3)
y_val_cat = keras.utils.to_categorical(y_style_val, 3)
y_test_cat = keras.utils.to_categorical(y_style_test, 3)

print(f"\n✅ Datos cargados: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")

# Crear modelo
def crear_modelo_control_calidad():
    model = models.Sequential([
        layers.Input(shape=(6,)),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(3, activation='softmax')
    ], name='Control_Calidad_Model')
    
    return model

modelo = crear_modelo_control_calidad()
modelo.summary()

# Compilar
modelo.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20),
    callbacks.ModelCheckpoint('modelos/modelo_1_control_calidad.h5', save_best_only=True)
]

# Entrenar
print("\n🚀 Entrenando Modelo 1...")
history = modelo.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=500,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluar
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_style_test, y_pred_classes)
cm = confusion_matrix(y_style_test, y_pred_classes)

print("\n" + "="*70)
print("📊 RESULTADOS MODELO 1")
print("="*70)
print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
print("\nMatriz de Confusión:")
print(cm)

# Guardar métricas
metricas = {
    'accuracy': float(accuracy),
    'confusion_matrix': cm.tolist(),
    'epochs_trained': len(history.history['loss'])
}

with open('modelos/modelo_1_metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)

with open('modelos/modelo_1_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

print("\n💾 Modelo 1 guardado: modelos/modelo_1_control_calidad.h5")
print("="*70)