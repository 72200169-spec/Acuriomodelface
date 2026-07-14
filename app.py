import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import math

# Configuración de la página
st.set_page_config(
    page_title="El Filósofo Virtual 🧠",
    page_icon="📜",
    layout="wide"
)

# Título principal
st.title("📜 El Filósofo Virtual")
st.markdown("---")

# Cargar el modelo de emociones
@st.cache_resource
def cargar_modelo():
    rutas_modelo = [
        'mejor_modelo.keras',
        './mejor_modelo.keras',
        '/mount/src/acuriomodelface/mejor_modelo.keras'
    ]
    for ruta in rutas_modelo:
        try:
            modelo = load_model(ruta)
            return modelo
        except Exception as e:
            continue
    st.error("❌ No se encontró el modelo!")
    return None

# Constantes y datos
EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOCIONES_ES = {
    'angry': 'Enojo',
    'disgust': 'Disgusto', 
    'fear': 'Miedo',
    'happy': 'Alegría',
    'neutral': 'Neutral',
    'sad': 'Tristeza',
    'surprise': 'Sorpresa'
}
IMG_SIZE = 48

# Citas filosóficas por emoción
CITAS_FILOSOFICAS = {
    'angry': [
        {"filosofo": "Aristóteles", "cita": "Cualquiera puede enojarse — eso es fácil. Pero enojarse con la persona correcta, en el grado correcto, en el momento correcto, por el motivo correcto y de la manera correcta, eso no es fácil."},
        {"filosofo": "Nietzsche", "cita": "Resentimiento es la venganza del débil."}
    ],
    'fear': [
        {"filosofo": "Sartre", "cita": "La libertad es lo que haces de lo que se ha hecho de ti."},
        {"filosofo": "Kant", "cita": "Sapere aude! ¡Atreve a saber!"}
    ],
    'happy': [
        {"filosofo": "Aristóteles", "cita": "La felicidad es el significado y el propósito de la vida, el objetivo y el fin de la existencia humana."},
        {"filosofo": "Nietzsche", "cita": "¡Caminemos por nuestro propio camino!"}
    ],
    'sad': [
        {"filosofo": "Sartre", "cita": "La existencia precede a la esencia."},
        {"filosofo": "Aristóteles", "cita": "El hombre es un ser social."}
    ],
    'surprise': [
        {"filosofo": "Nietzsche", "cita": "¡Qué importan los momentos! ¡La vida es un ritmo, una danza, una fiesta!"},
        {"filosofo": "Kant", "cita": "La curiosidad es el motor del pensamiento."}
    ],
    'neutral': [
        {"filosofo": "Kant", "cita": "No hagas a los demás lo que no quieres que te hagan a ti."},
        {"filosofo": "Sartre", "cita": "Estamos condenados a ser libres."}
    ],
    'disgust': [
        {"filosofo": "Aristóteles", "cita": "La virtud es un término medio entre dos vicios."}
    ]
}

# Cargar modelo
modelo = cargar_modelo()

# --- Funciones auxiliares ---

# 1. Emociones
def preprocesar_imagen(imagen):
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    imagen_redimensionada = cv2.resize(imagen_gris, (IMG_SIZE, IMG_SIZE))
    imagen_normalizada = imagen_redimensionada / 255.0
    return imagen_normalizada.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def detectar_rostro(imagen):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if len(imagen.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    rostros = face_cascade.detectMultiScale(imagen_gris, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    if len(rostros) > 0:
        (x, y, w, h) = rostros[0]
        return imagen_gris[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def predecir_emocion(imagen):
    if modelo is None:
        return None, None
    img_procesada = preprocesar_imagen(imagen)
    pred = modelo.predict(img_procesada, verbose=0)[0]
    return EMOCIONES[np.argmax(pred)], {EMOCIONES[i]: float(pred[i]) for i in range(len(EMOCIONES))}

# 2. Gestos con MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def obtener_distancia(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def detectar_gesto(landmarks, image_width, image_height):
    points = []
    for lm in landmarks.landmark:
        points.append([lm.x * image_width, lm.y * image_height])
    
    wrist = points[0]
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    ring_tip = points[16]
    pinky_tip = points[20]
    thumb_ip = points[3]
    index_pip = points[6]
    middle_pip = points[10]
    ring_pip = points[14]
    pinky_pip = points[18]
    
    # Pulgar arriba
    thumb_up = thumb_tip[1] < thumb_ip[1] and thumb_tip[1] < index_tip[1] and all(
        tip[1] > pip[1] for tip, pip in [(index_tip, index_pip), (middle_tip, middle_pip), (ring_tip, ring_pip), (pinky_tip, pinky_pip)]
    )
    
    # Mano abierta (todos los dedos extendidos)
    open_hand = all(tip[1] < pip[1] for tip, pip in [
        (thumb_tip, thumb_ip), (index_tip, index_pip), (middle_tip, middle_pip), 
        (ring_tip, ring_pip), (pinky_tip, pinky_pip)
    ])
    
    # OK (pulgar e índice juntos)
    ok_dist = obtener_distancia(thumb_tip, index_tip) < 50
    
    if open_hand:
        return "MANO ABIERTA"
    elif thumb_up:
        return "PULGAR ARRIBA"
    elif ok_dist:
        return "OK"
    else:
        return None

# --- Barra lateral ---
st.sidebar.title("⚙️ Modos")
modo = st.sidebar.radio(
    "Selecciona el modo de operación:",
    ["Modo Filósofo Emocional 😊", "Modo Gestos 👋"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tips: Ilumina bien tu rostro para detectar emociones. Para gestos, coloca tu mano frente a la cámara.")

# --- Contenido principal ---
if modo == "Modo Filósofo Emocional 😊":
    st.header("🧘 Modo Filósofo Emocional")
    
    foto = st.camera_input("Captura una foto de tu rostro")
    
    if foto is not None:
        img_pil = Image.open(foto)
        img_array = np.array(img_pil)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_pil, caption="Tu foto", use_container_width=True)
        
        with col2:
            if modelo is None:
                st.error("❌ No hay modelo")
            else:
                rostro, coords = detectar_rostro(img_array)
                if rostro is not None:
                    emocion, probs = predecir_emocion(rostro)
                    
                    # Mostrar emoción
                    emoji_map = {'angry': '😠', 'fear': '😨', 'happy': '😊', 'sad': '😢', 'surprise': '😮', 'neutral': '😐', 'disgust': '🤢'}
                    st.success(f"🎯 Emoción: **{EMOCIONES_ES[emocion]}** {emoji_map.get(emocion, '')}")
                    
                    # Mostrar cita filosófica
                    if emocion in CITAS_FILOSOFICAS:
                        cita = CITAS_FILOSOFICAS[emocion][np.random.randint(0, len(CITAS_FILOSOFICAS[emocion]))]
                        st.markdown(f"""
                        > *\"{cita['cita']}\"*
                        > 
                        > — **{cita['filosofo']}**
                        """)
                else:
                    st.warning("⚠️ No se detectó un rostro")

elif modo == "Modo Gestos 👋":
    st.header("🖐️ Modo Gestos")
    
    foto = st.camera_input("Captura una foto de tu mano")
    
    if foto is not None:
        img_pil = Image.open(foto)
        img_array = np.array(img_pil)
        h, w, _ = img_array.shape
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_pil, caption="Tu mano", use_container_width=True)
        
        with col2:
            with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesto = detectar_gesto(hand_landmarks, w, h)
                        
                        if gesto:
                            st.success(f"🎯 Gesto detectado: **{gesto}**")
                            
                            # Interacción según gesto
                            if gesto == "MANO ABIERTA":
                                st.info("👋 ¡Hola! Bienvenido al Filósofo Virtual")
                            elif gesto == "PULGAR ARRIBA":
                                st.info("👍 ¡Me alegra que estés aquí!")
                            elif gesto == "OK":
                                st.info("👌 ¡Perfecto!")
                else:
                    st.warning("⚠️ No se detectó ninguna mano")

# --- Pie de página ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, TensorFlow and MediaPipe")
