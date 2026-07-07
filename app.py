import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Expresiones Faciales 😊",
    page_icon="🎭",
    layout="wide"
)

# Título principal
st.title("🎭 Reconocimiento de Expresiones Faciales")
st.markdown("---")

# Cargar el modelo y definir variables globales
@st.cache_resource
def cargar_modelo():
    try:
        # Intentar cargar el modelo desde el directorio actual
        modelo = load_model('mejor_modelo.keras')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("Primero entrena el modelo con train.py o asegúrate de que 'mejor_modelo.keras' esté en el mismo directorio!")
        return None

# Lista de emociones (en orden alfabético, como lo usa ImageDataGenerator)
EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = 48

# Cargar modelo
modelo = cargar_modelo()

# Función para preprocesar la imagen
def preprocesar_imagen(imagen):
    # Convertir a escala de grises
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    
    # Redimensionar a 48x48
    imagen_redimensionada = cv2.resize(imagen_gris, (IMG_SIZE, IMG_SIZE))
    
    # Normalizar y dar forma para el modelo
    imagen_normalizada = imagen_redimensionada / 255.0
    imagen_final = imagen_normalizada.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return imagen_final

# Función para detectar rostro con OpenCV
def detectar_rostro(imagen):
    # Cargar el clasificador de rostros de OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convertir a escala de grises
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    
    # Detectar rostros (más flexible: scaleFactor más bajo, minNeighbors más bajo)
    rostros = face_cascade.detectMultiScale(
        imagen_gris, 
        scaleFactor=1.05,  # Menor = más sensible
        minNeighbors=3,    # Menor = más detecciones (aunque puede haber falsos positivos)
        minSize=(30, 30)
    )
    
    if len(rostros) > 0:
        # Tomar el primer rostro
        (x, y, w, h) = rostros[0]
        rostro_recortado = imagen_gris[y:y+h, x:x+w]
        return rostro_recortado, (x, y, w, h)
    else:
        return None, None

# Función para hacer la predicción
def predecir_emocion(imagen):
    if modelo is None:
        return None, None
    
    imagen_procesada = preprocesar_imagen(imagen)
    prediccion = modelo.predict(imagen_procesada, verbose=0)[0]
    emocion_predicha = EMOCIONES[np.argmax(prediccion)]
    probabilidades = {EMOCIONES[i]: float(prediccion[i]) for i in range(len(EMOCIONES))}
    
    return emocion_predicha, probabilidades

# Función para mostrar gráfico de barras
def mostrar_grafico_probabilidades(probabilidades):
    fig, ax = plt.subplots(figsize=(10, 5))
    emociones_lista = list(probabilidades.keys())
    valores = list(probabilidades.values())
    
    # Colores bonitos
    colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    ax.bar(emociones_lista, valores, color=colores)
    ax.set_title('📊 Probabilidades de cada emoción', fontsize=14)
    ax.set_xlabel('Emoción', fontsize=12)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Barra lateral
st.sidebar.title("⚙️ Opciones")
opcion = st.sidebar.radio(
    "Selecciona una opción:",
    ["Subir una imagen 📷", "Usar Cámara Web 🎥"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tips: Asegúrate de que la imagen tenga un rostro bien iluminado y centrado para mejores resultados!")

# Contenido principal
if opcion == "Subir una imagen 📷":
    st.header("📷 Subir una imagen")
    
    archivo_subido = st.file_uploader("Selecciona una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if archivo_subido is not None:
        # Cargar la imagen
        imagen = Image.open(archivo_subido)
        imagen_array = np.array(imagen)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Imagen original")
            st.image(imagen, width=400)
        
        with col2:
            st.subheader("Resultado")
            
            # Primero, verificamos si el modelo está cargado
            if modelo is None:
                st.error("❌ No hay modelo cargado! Asegúrate de que 'mejor_modelo.keras' esté en el directorio.")
            else:
                # Detectar rostro
                rostro, coordenadas = detectar_rostro(imagen_array)
                
                if rostro is not None:
                    st.info("✅ Rostro detectado! Analizando...")
                    
                    # Predecir emoción
                    emocion, probabilidades = predecir_emocion(rostro)
                    
                    if emocion is not None:
                        # Emoji por emoción
                        emojis = {
                            'angry': '😠',
                            'disgust': '🤢',
                            'fear': '😨',
                            'happy': '😊',
                            'neutral': '😐',
                            'sad': '😢',
                            'surprise': '😮'
                        }
                        
                        st.success(f"🎯 Emoción detectada: **{emocion.upper()}** {emojis.get(emocion, '')}")
                        
                        # Mostrar gráfico
                        st.subheader("Probabilidades detalladas")
                        fig = mostrar_grafico_probabilidades(probabilidades)
                        st.pyplot(fig)
                else:
                    st.warning("⚠️ No se detectó ningún rostro en la imagen. Intenta con otra foto donde el rostro esté bien iluminado y centrado!")

elif opcion == "Usar Cámara Web 🎥":
    st.header("🎥 Usar Cámara Web")
    
    foto_capturada = st.camera_input("Captura una foto con tu cámara")
    
    if foto_capturada is not None:
        # Cargar la imagen
        imagen = Image.open(foto_capturada)
        imagen_array = np.array(imagen)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Foto capturada")
            st.image(imagen, width=400)
        
        with col2:
            st.subheader("Resultado")
            
            # Primero, verificamos si el modelo está cargado
            if modelo is None:
                st.error("❌ No hay modelo cargado! Asegúrate de que 'mejor_modelo.keras' esté en el directorio.")
            else:
                # Detectar rostro
                rostro, coordenadas = detectar_rostro(imagen_array)
                
                if rostro is not None:
                    st.info("✅ Rostro detectado! Analizando...")
                    
                    # Predecir emoción
                    emocion, probabilidades = predecir_emocion(rostro)
                    
                    if emocion is not None:
                        # Emoji por emoción
                        emojis = {
                            'angry': '😠',
                            'disgust': '🤢',
                            'fear': '😨',
                            'happy': '😊',
                            'neutral': '😐',
                            'sad': '😢',
                            'surprise': '😮'
                        }
                        
                        st.success(f"🎯 Emoción detectada: **{emocion.upper()}** {emojis.get(emocion, '')}")
                        
                        # Mostrar gráfico
                        st.subheader("Probabilidades detalladas")
                        fig = mostrar_grafico_probabilidades(probabilidades)
                        st.pyplot(fig)
                else:
                    st.warning("⚠️ No se detectó ningún rostro en la foto. Intenta nuevamente con buena iluminación!")

# Pie de página
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, TensorFlow and OpenCV")
