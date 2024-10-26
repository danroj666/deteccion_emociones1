from flask import Flask, render_template, request, redirect, jsonify
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
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')

# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configuración de MediaPipe para detección de rostros
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def analizar_imagen(ruta_imagen):
    """
    Analiza una imagen y guarda una nueva imagen con puntos faciales clave detectados.
    """
    # Leer y procesar la imagen
    imagen = cv2.imread(ruta_imagen)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(imagen_rgb)
    
    if not resultados.multi_face_landmarks:
        raise Exception("No se detectó rostro en la imagen")

    # Selección de puntos clave principales
    puntos_clave = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
    
    # Obtener dimensiones de la imagen
    altura, anchura, _ = imagen.shape
    
    # Dibujar puntos en la imagen
    for punto_idx in puntos_clave:
        landmark = resultados.multi_face_landmarks[0].landmark[punto_idx]
        x = int(landmark.x * anchura)
        y = int(landmark.y * altura)
        cv2.drawMarker(
            imagen, (x, y), (0, 0, 255),  # Color en formato BGR
            markerType=cv2.MARKER_CROSS,
            markerSize=15,  # Tamaño del marcador
            thickness=5     # Grosor de la línea del marcador
        )

    # Guardar la imagen procesada
    resultado_path = os.path.join(app.config['UPLOAD_FOLDER'], "resultado.jpg")
    cv2.imwrite(resultado_path, imagen)
    return resultado_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Verificar si se subió un archivo
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Guardar y analizar la imagen
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                resultado_path = analizar_imagen(file_path)
                return render_template('index.html', resultado=resultado_path)
            except Exception as e:
                return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == '__main__':
    # Iniciar un túnel ngrok en el puerto 5000
    public_url = ngrok.connect(5000)
    print(f" * ngrok URL: {public_url}")

    # Ejecuta Flask en el puerto 5000
    app.run(port=5000)
