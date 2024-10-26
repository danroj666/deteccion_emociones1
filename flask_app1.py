from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import io
import os
import cv2
import numpy as np
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
from pyngrok import ngrok
import mediapipe as mp

# Crear la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Configuración de MediaPipe para detección de rostros
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def analizar_imagen(img_array):
    """
    Analiza una imagen y retorna una nueva imagen con puntos faciales clave detectados en formato base64.
    """
    # Convertir la imagen a RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    # Dibujar los puntos en la imagen
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)

    # Selección de puntos clave principales
    puntos_clave = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
    for landmark in results.multi_face_landmarks:
        for idx in puntos_clave:
            x = int(landmark.landmark[idx].x * img_array.shape[1])
            y = int(landmark.landmark[idx].y * img_array.shape[0])
            ax.plot(x, y, 'm+', markersize=15)

    # Guardar la imagen generada en memoria
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    output.seek(0)

    # Convertir la imagen a base64
    encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
    return encoded_image

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

    # Procesar la imagen y detectar rostros
    encoded_image = analizar_imagen(img)

    if not encoded_image:
        return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

    return jsonify({'image': encoded_image})

# Ruta para servir los archivos subidos
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ejecuta la aplicación Flask
if __name__ == '__main__':
    # Iniciar un túnel ngrok en el puerto 5000
    public_url = ngrok.connect(5000)
    print(f" * ngrok URL: {public_url}")

    # Ejecuta Flask en el puerto 5000
    app.run(port=5000)
