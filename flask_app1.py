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

# Crear la aplicación Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')


# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Cargar el clasificador de Haar para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para procesar y generar la imagen con puntos clave en el rostro
def generate_image_with_keypoints(image_array, faces):
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image_array, cmap='gray')  # Mostrar la imagen subida en escala de grises

    for (x, y, w, h) in faces:
        reduced_x = x + int(w * 0.2)  # Recortar el 20% de los bordes laterales
        reduced_y = y + int(h * 0.2)  # Recortar el 20% desde arriba (excluir cabello)
        reduced_w = int(w * 0.6)  # Solo el 60% de la anchura central
        reduced_h = int(h * 0.6)  # Solo el 60% de la altura (centrada en el rostro)

        num_points = 15  # Número de puntos clave (ajustable)

        for _ in range(num_points):
            point_x = np.random.randint(reduced_x, reduced_x + reduced_w)
            point_y = np.random.randint(reduced_y, reduced_y + reduced_h)

            plt.plot(point_x, point_y, 'm+', markersize=15)  # Dibuja el punto en morado y más grande

    # Guardar la imagen generada en memoria
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    output.seek(0)

    return output

# Función para aumentar el brillo de la imagen
def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar el rostro en la imagen
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

    # Aumentar el brillo de la imagen
    bright_img = increase_brightness(img)

    # Voltear la imagen
    flipped_img = cv2.flip(bright_img, 0)  # 0 para voltear de cabeza

    # Generar la imagen con puntos clave en el rostro
    output = generate_image_with_keypoints(flipped_img, faces)

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
