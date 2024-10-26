from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
import io
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
from pyngrok import ngrok
import base64
import mediapipe as mp

# Crear la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configuración de MediaPipe para detección de rostros
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Función para procesar y generar la imagen con puntos faciales clave
def generate_image_with_keypoints_mediapipe(image_array):
    # Convertir la imagen a RGB para usar en MediaPipe
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise Exception("No se detectó rostro en la imagen")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))

    # Selección de puntos clave principales
    puntos_clave = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
    altura, anchura, _ = image_array.shape
    
    for punto_idx in puntos_clave:
        landmark = results.multi_face_landmarks[0].landmark[punto_idx]
        x = int(landmark.x * anchura)
        y = int(landmark.y * altura)
        ax.plot(x, y, 'go', markersize=8)  # Dibuja puntos en verde
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    output.seek(0)
    
    return output

# Página principal con el formulario para subir imágenes
@app.route('/')
def index():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', images=images)

# Ruta para subir y analizar la imagen
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        img = cv2.imread(filepath)
    elif 'existing_file' in request.form:
        filename = request.form['existing_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = cv2.imread(filepath)
    else:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

    try:
        output = generate_image_with_keypoints_mediapipe(img)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
    return jsonify({'image': encoded_image})

# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" * ngrok URL: {public_url}")
    app.run(port=5000)
