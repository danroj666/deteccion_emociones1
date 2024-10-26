from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
import io
import os
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pyngrok import ngrok
import base64
import mediapipe as mp

# Crear la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuración de MediaPipe para detección de puntos faciales
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def analizar_imagen(image_array):
    """
    Analiza una imagen y genera una nueva imagen con puntos faciales clave detectados.
    """
    # Procesar la imagen en formato RGB
    imagen_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(imagen_rgb)
    
    if not resultados.multi_face_landmarks:
        raise Exception("No se detectó rostro en la imagen")

    # Selección de puntos clave específicos
    puntos_clave = [33, 133, 362, 263, 1, 61, 291, 199, 94, 0, 24, 130, 359, 288, 378]
    
    # Obtener dimensiones de la imagen
    altura, anchura, _ = image_array.shape
    
    # Dibujar puntos en la imagen con tamaño, grosor y color personalizados
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))  # Mostrar la imagen en color

    for punto_idx in puntos_clave:
        landmark = resultados.multi_face_landmarks[0].landmark[punto_idx]
        x = int(landmark.x * anchura)
        y = int(landmark.y * altura)
        ax.plot(x, y, 'm+', markersize=15)  # Dibujar puntos en morado

    # Guardar la imagen generada en memoria
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

    # Procesar y generar la imagen con puntos clave
    try:
        output = analizar_imagen(img)
        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return jsonify({'image': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
