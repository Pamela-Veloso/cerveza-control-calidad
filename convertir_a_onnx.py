import tensorflow as tf
import tf2onnx
import onnx

print("="*70)
print("🔄 CONVIRTIENDO MODELOS A ONNX")
print("="*70)

# Modelo 1
print("\n📦 Convirtiendo Modelo 1...")
modelo_1 = tf.keras.models.load_model('modelos/modelo_1_control_calidad.h5')
onnx_model_1, _ = tf2onnx.convert.from_keras(modelo_1)
onnx.save(onnx_model_1, 'modelos/modelo_1.onnx')
print("✅ Modelo 1 convertido")

# Modelo 2
print("\n📦 Convirtiendo Modelo 2...")
modelo_2 = tf.keras.models.load_model('modelos/modelo_2_predictor_abv.h5')
onnx_model_2, _ = tf2onnx.convert.from_keras(modelo_2)
onnx.save(onnx_model_2, 'modelos/modelo_2.onnx')
print("✅ Modelo 2 convertido")

# Modelo 3
print("\n📦 Convirtiendo Modelo 3...")
modelo_3 = tf.keras.models.load_model('modelos/modelo_3_clasificador_experimental.h5')
onnx_model_3, _ = tf2onnx.convert.from_keras(modelo_3)
onnx.save(onnx_model_3, 'modelos/modelo_3.onnx')
print("✅ Modelo 3 convertido")

print("\n" + "="*70)
print("✅ TODOS LOS MODELOS CONVERTIDOS A ONNX")
print("="*70)