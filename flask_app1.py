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
app.config['DIRECTORIO_SUBIDAS'] = os.path.join(app.root_path, 'uploads')

# Crear el directorio para subir archivos si no existe
if not os.path.exists(app.config['DIRECTORIO_SUBIDAS']):
    os.makedirs(app.config['DIRECTORIO_SUBIDAS'])

# Cargar el clasificador de Haar para detección de rostros
clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para procesar y generar la imagen con puntos clave en el rostro
def crear_imagen_con_puntos(imagen_array, rostros):
    figura = plt.figure(figsize=(20, 20))
    plt.imshow(imagen_array, cmap='gray')  # Mostrar la imagen subida en escala de grises

    for (x, y, w, h) in rostros:
        recorte_x = x + int(w * 0.2)  # Recortar el 20% de los bordes laterales
        recorte_y = y + int(h * 0.2)  # Recortar el 20% desde arriba (excluir cabello)
        recorte_w = int(w * 0.6)  # Solo el 60% de la anchura central
        recorte_h = int(h * 0.6)  # Solo el 60% de la altura (centrada en el rostro)

        num_puntos = 15  # Número de puntos clave (ajustable)

        for _ in range(num_puntos):
            punto_x = np.random.randint(recorte_x, recorte_x + recorte_w)
            punto_y = np.random.randint(recorte_y, recorte_y + recorte_h)

            plt.plot(punto_x, punto_y, 'ro', markersize=20)  # Dibuja el punto en rojo y más grande

    # Guardar la imagen generada en memoria
    salida = io.BytesIO()
    FigureCanvas(figura).print_png(salida)
    plt.close(figura)
    salida.seek(0)

    return salida

# Página principal con el formulario para subir imágenes
@app.route('/')
def pagina_principal():
    # Obtener la lista de archivos subidos
    imagenes = os.listdir(app.config['DIRECTORIO_SUBIDAS'])
    return render_template('index.html', images=imagenes)

# Ruta para subir y analizar la imagen
@app.route('/analizar', methods=['POST'])
def analizar_imagen():
    # Verificar si se subió una imagen nueva o si se seleccionó una existente
    if 'archivo' in request.files:
        archivo = request.files['archivo']
        if archivo.filename == '':
            return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

        nombre_archivo = secure_filename(archivo.filename)
        ruta_archivo = os.path.join(app.config['DIRECTORIO_SUBIDAS'], nombre_archivo)
        archivo.save(ruta_archivo)
        img = cv2.imread(ruta_archivo)
    elif 'archivo_existente' in request.form:
        nombre_archivo = request.form['archivo_existente']
        ruta_archivo = os.path.join(app.config['DIRECTORIO_SUBIDAS'], nombre_archivo)
        img = cv2.imread(ruta_archivo)
    else:
        return jsonify({'error': 'No se ha proporcionado ninguna imagen.'}), 400

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar el rostro en la imagen
    rostros = clasificador_rostros.detectMultiScale(imagen_gris, scaleFactor=1.1, minNeighbors=5)

    if len(rostros) == 0:
        return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

    # Generar la imagen con puntos clave en el rostro
    salida_imagen = crear_imagen_con_puntos(imagen_gris, rostros)

    # Convertir la imagen generada a base64 para enviarla en la respuesta
    imagen_codificada = base64.b64encode(salida_imagen.getvalue()).decode('utf-8')

    return jsonify({'image': imagen_codificada})

# Ruta para servir los archivos subidos
@app.route('/uploads/<nombre_archivo>')
def archivo_subido(nombre_archivo):
    return send_from_directory(app.config['DIRECTORIO_SUBIDAS'], nombre_archivo)

# Ejecuta la aplicación Flask
if __name__ == '__main__':
    # Iniciar un túnel ngrok en el puerto 5000
    url_publica = ngrok.connect(5000)
    print(f" * ngrok URL: {url_publica}")

    # Ejecuta Flask en el puerto 5000
    app.run(port=5000)
