from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
from werkzeug.utils import secure_filename

# ========================
# Inisialisasi Flask
# ========================
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ========================
# Load Model
# ========================
model = tf.keras.models.load_model("model_mobilenetv2_ras_kucing.keras")

# ========================
# Daftar Kelas (10 Kelas)
# ========================
class_names = [
    'American Shorthair',
    'Bengal',
    'British Shorthair',
    'Domestic Short Hair',
    'Maine Coon',
    'Munchkin',
    'Persian',
    'Ragdoll',
    'Scottish Fold',
    'Sphynx - Hairless Cat'
]

# ========================
# Validasi Ekstensi File
# ========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========================
# Fungsi Prediksi Gambar
# ========================
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    return class_names[predicted_index], confidence

# ========================
# Halaman Utama
# ========================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            # Simpan file dengan nama unik
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Prediksi Gambar
            prediction, confidence = predict_image(file_path)

            # Path relatif ke folder static
            image_url = url_for('static', filename=f'uploads/{filename}')

            return render_template('index.html',
                                   prediction=prediction,
                                   confidence=round(confidence, 2),
                                   image_path=image_url)
        else:
            return render_template('index.html',
                                   prediction="Format file tidak valid. Gunakan JPG/PNG.",
                                   confidence=0,
                                   image_path=None)

    return render_template('index.html')

# ========================
# Jalankan Aplikasi
# ========================
if __name__ == '__main__':
    app.run(debug=True)
