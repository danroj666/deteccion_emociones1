from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from pyngrok import ngrok

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Crear el directorio para subir archivos si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuración de MediaPipe para detección de puntos faciales clave
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Puntos faciales clave principales
puntos_clave = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]

def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen y genera una nueva con puntos faciales clave detectados.
    """
    # Leer y procesar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(imagen_rgb)
    
    if not resultados.multi_face_landmarks:
        raise Exception("No se detectó rostro en la imagen")

    altura, anchura, _ = imagen.shape

    # Dibujar puntos en la imagen
    for punto_idx in puntos_clave:
        landmark = resultados.multi_face_landmarks[0].landmark[punto_idx]
        x = int(landmark.x * anchura)
        y = int(landmark.y * altura)
        cv2.drawMarker(
            imagen, (x, y), (255, 0, 0),  # Color en formato BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=15,
            thickness=5
        )

    # Convertir la imagen procesada a base64
    _, buffer = cv2.imencode('.jpg', imagen)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', images=images)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    elif 'existing_file' in request.form:
        filename = request.form['existing_file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

    try:
        # Analizar y generar la imagen procesada con puntos faciales clave
        encoded_image = analizar_imagen(filepath)
        return jsonify({'image': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Iniciar un túnel ngrok en el puerto 5000
    public_url = ngrok.connect(5000)
    print(f" * ngrok URL: {public_url}")

    # Ejecuta Flask en el puerto 5000
    app.run(port=5000)
