### machine_learning_engineer_technical-_test
Develop a machine learning model to forecast the closing price of a single Indonesian stock for the next day
Machine Learning Model with MLflow Tracking

Proyek ini berisi kode untuk melatih model machine learning dengan tracking menggunakan MLflow dan menjalankan prediksi melalui API dengan Uvicorn.

## Instalasi

Pastikan Python sudah terinstall (versi 3.8+ disarankan), lalu jalankan perintah berikut untuk menginstall dependensi:

pip install -r requirements.txt


Jika file requirements.txt belum dibuat, kamu bisa langsung install MLflow:

pip install mlflow

## Training Model

Untuk melatih model dan melacak eksperimen menggunakan MLflow, jalankan:

python train_model.py


Setelah training, kamu bisa membuka UI MLflow untuk melihat hasil eksperimen:

mlflow ui


Lalu buka browser di:

http://127.0.0.1:5000

## Menjalankan API Prediksi

Setelah model dilatih, jalankan API menggunakan Uvicorn:

uvicorn predict_web:app --reload


API akan berjalan di:

http://127.0.0.1:8000
