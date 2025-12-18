import joblib
import numpy as np

model = joblib.load('model.pkl') 

# Sesuaikan urutan angkanya: Type, Air temp, Process temp, RPM, Torque, Tool wear
data = np.array([[1, 298.1, 308.6, 1551, 42.8, 0]])

prediction = model.predict(data)

print(f"--- HASIL INFERENCE BERHASIL ---")
print(f"Hasil Prediksi: {prediction}")