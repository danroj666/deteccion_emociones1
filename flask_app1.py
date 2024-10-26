from flask import Flask, render_template, request, jsonify, send_from_directory
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
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Función para procesar y generar la imagen con puntos clave en el rostro
def generate_image_with_keypoints(image_array, landmarks):
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image_array, cmap='gray')

    # Dibujar cada punto clave en la imagen
    altura, anchura = image_array.shape
    for landmark in landmarks:
        x = int(landmark.x * anchura)
        y = int(landmark.y * altura)
        plt.plot(x, y, 'm+', markersize=15)  # Dibuja el punto en morado y más grande

    # Guardar la imagen generada en memoria
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    output.seek(0)

    return output

# Página principal con el formulario para subir imágenes
@app.route('/')
def index():
    # Obtener la lista de archivos subidos
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', images=images)

# Ruta para subir y analizar la imagen
@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Verificar si se subió una imagen nueva o si se seleccionó una existente
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

    # Convertir la imagen a escala de grises
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detectar el rostro y los puntos clave en la imagen
    resultados = face_mesh.process(img_rgb)

    if not resultados.multi_face_landmarks:
        return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

    # Tomar los puntos faciales de la primera cara detectada
    landmarks = resultados.multi_face_landmarks[0].landmark

    # Generar la imagen con puntos clave en el rostro
    output = generate_image_with_keypoints(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), landmarks)

    # Convertir la imagen generada a base64 para enviarla en la respuesta
    encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')

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
