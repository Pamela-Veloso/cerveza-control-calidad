"""
MODELO 3: CLASIFICADOR EXPERIMENTAL
Similar al Modelo 1 pero enfocado en dar probabilidades detalladas
Input: OG, ABV, pH, IBU
Output: Estilo + probabilidades de cada estilo
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
import json

print("="*70)
print("🍺 MODELO 3: CLASIFICADOR EXPERIMENTAL")
print("="*70)

# Reutilizar datos del Modelo 1
X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')
X_test = np.load('data/X_test.npy')

y_style_train = np.load('data/y_style_train.npy')
y_style_val = np.load('data/y_style_val.npy')
y_style_test = np.load('data/y_style_test.npy')

y_train_cat = keras.utils.to_categorical(y_style_train, 3)
y_val_cat = keras.utils.to_categorical(y_style_val, 3)
y_test_cat = keras.utils.to_categorical(y_style_test, 3)

print(f"\n✅ Datos cargados")

# Crear modelo (arquitectura ligeramente diferente)
def crear_modelo_experimental():
    model = models.Sequential([
        layers.Input(shape=(6,)),
        
        layers.Dense(96, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Dense(48, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        
        layers.Dense(24, activation='relu'),
        layers.Dropout(0.25),
        
        layers.Dense(3, activation='softmax')
    ], name='Clasificador_Experimental_Model')
    
    return model

modelo = crear_modelo_experimental()
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
    callbacks.ModelCheckpoint('modelos/modelo_3_clasificador_experimental.h5', save_best_only=True)
]

# Entrenar
print("\n🚀 Entrenando Modelo 3...")
history = modelo.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=500,
    batch_size=16,
    callbacks=callbacks_list,
    verbose=1
)

# Evaluar - enfoque en probabilidades
y_pred_proba = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred_proba, axis=1)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("\n" + "="*70)
print("📊 RESULTADOS MODELO 3")
print("="*70)

print("\n🔍 Ejemplos de predicciones con probabilidades:")
for i in range(min(10, len(y_test_cat))):
    real = label_encoder.classes_[y_style_test[i]]
    pred = label_encoder.classes_[y_pred_classes[i]]
    probas = y_pred_proba[i]
    
    print(f"\nMuestra {i+1}: Real={real}, Predicho={pred}")
    for j, style in enumerate(label_encoder.classes_):
        print(f"   {style}: {probas[j]*100:.1f}%")

# Reporte
print("\n" + classification_report(y_style_test, y_pred_classes, 
                                    target_names=label_encoder.classes_))

# Guardar
metricas = {
    'accuracy': float(np.mean(y_style_test == y_pred_classes)),
    'epochs_trained': len(history.history['loss'])
}

with open('modelos/modelo_3_metricas.json', 'w') as f:
    json.dump(metricas, f, indent=2)

with open('modelos/modelo_3_history.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

np.save('modelos/modelo_3_y_pred_proba.npy', y_pred_proba)

print("\n💾 Modelo 3 guardado: modelos/modelo_3_clasificador_experimental.h5")
print("="*70)