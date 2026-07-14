import math
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

    REALTIME_AVAILABLE = True
    REALTIME_IMPORT_ERROR = ""
except Exception as exc:
    REALTIME_AVAILABLE = False
    REALTIME_IMPORT_ERROR = str(exc)


st.set_page_config(
    page_title="El Filosofo Virtual",
    page_icon="📜",
    layout="wide",
)

EMOCIONES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOCIONES_ES = {
    "angry": "Enojo",
    "disgust": "Disgusto",
    "fear": "Miedo",
    "happy": "Alegria",
    "neutral": "Neutral",
    "sad": "Tristeza",
    "surprise": "Sorpresa",
}
EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😮",
}
IMG_SIZE = 48
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

CONTENIDO_FILOSOFICO = {
    "happy": {
        "filosofo": "Aristoteles",
        "frase": "La felicidad es el sentido y el proposito de la vida.",
        "pregunta": "¿Que habito bueno te gustaria sostener para que esta alegria no sea pasajera?",
        "accion": "Aprovecha este estado para reforzar una rutina saludable o agradecer algo concreto.",
    },
    "sad": {
        "filosofo": "Sartre",
        "frase": "La existencia precede a la esencia.",
        "pregunta": "¿Que pequena decision puedes tomar hoy para no quedarte atrapado en este momento?",
        "accion": "Respira, baja el ritmo y convierte el malestar en una accion corta y posible.",
    },
    "angry": {
        "filosofo": "Aristoteles",
        "frase": "Enojarse con la persona correcta, en el grado correcto y en el momento correcto no es facil.",
        "pregunta": "¿Tu energia esta defendiendo un valor importante o solo esta reaccionando por impulso?",
        "accion": "Haz una pausa de 10 segundos antes de responder. La claridad vale mas que la descarga.",
    },
    "surprise": {
        "filosofo": "Nietzsche",
        "frase": "Hay que llevar todavia un caos dentro de si para poder dar a luz una estrella danzante.",
        "pregunta": "¿Que oportunidad escondida hay en eso que te tomo por sorpresa?",
        "accion": "Convierte la sorpresa en curiosidad. Observa primero, interpreta despues.",
    },
    "fear": {
        "filosofo": "Kant",
        "frase": "Atrevete a saber.",
        "pregunta": "¿Que paso pequeno te acercaria a lo que hoy te da miedo?",
        "accion": "Divide el problema en una tarea minima y haz solo esa parte ahora.",
    },
    "disgust": {
        "filosofo": "Nietzsche",
        "frase": "Quien tiene un por que para vivir puede soportar casi cualquier como.",
        "pregunta": "¿Que valor tuyo se esta sintiendo rechazado o invadido en este momento?",
        "accion": "Alejate un momento, ordena la situacion y vuelve con un criterio claro.",
    },
    "neutral": {
        "filosofo": "Kant",
        "frase": "Obra de tal modo que tu conducta pueda convertirse en una regla universal.",
        "pregunta": "¿Que decision serena te haria sentir orgulloso de ti hoy?",
        "accion": "La calma tambien es una ventaja: usala para pensar con precision.",
    },
}

MENSAJES_GESTO = {
    "MANO ABIERTA": "Abriste el espacio de dialogo. Buena senal para iniciar una reflexion.",
    "PULGAR ARRIBA": "Tomado: mantengamos una linea de reflexion mas motivadora.",
    "OK": "Has confirmado el enfoque actual. Seguimos profundizando en esta idea.",
    "MANO LEVANTADA": "Parece que quieres intervenir. Te dejo una pregunta para profundizar.",
    "BRAZOS ABIERTOS": "Tu postura muestra apertura. Es un buen momento para explorar nuevas ideas.",
}


@st.cache_resource
def cargar_modelo_emociones():
    rutas = [
        "mejor_modelo.h5",
        "./mejor_modelo.h5",
        "/mount/src/acuriomodelface/mejor_modelo.h5",
        "mejor_modelo.keras",
    ]
    errores = []

    for ruta in rutas:
        if not os.path.exists(ruta):
            continue
        try:
            return load_model(ruta, compile=False), ruta, errores
        except Exception as exc:
            errores.append(f"{ruta}: {exc}")

    return None, None, errores


@st.cache_resource
def cargar_detector_rostro():
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade


MODELO, MODELO_PATH, ERRORES_MODELO = cargar_modelo_emociones()
FACE_CASCADE = cargar_detector_rostro()
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def preprocesar_rostro(face_gray):
    resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def predecir_emocion(face_gray):
    if MODELO is None:
        return None, {}

    prediction = MODELO.predict(preprocesar_rostro(face_gray), verbose=0)[0]
    probabilities = {EMOCIONES[i]: float(prediction[i]) for i in range(len(EMOCIONES))}
    emotion = EMOCIONES[int(np.argmax(prediction))]
    return emotion, probabilities


def detectar_gesto_mano(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y
    thumb_up = lm[4].y < lm[3].y

    hand_scale = math.hypot(lm[5].x - lm[17].x, lm[5].y - lm[17].y) + 1e-6
    ok_distance = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) / hand_scale

    if all([index_up, middle_up, ring_up, pinky_up, thumb_up]):
        return "MANO ABIERTA"
    if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
        return "PULGAR ARRIBA"
    if ok_distance < 0.35 and middle_up and ring_up and pinky_up:
        return "OK"
    return None


def detectar_gesto_pose(pose_landmarks):
    left_shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    if min(left_shoulder.visibility, right_shoulder.visibility, left_wrist.visibility, right_wrist.visibility) < 0.45:
        return None

    if left_wrist.y < left_shoulder.y - 0.05 or right_wrist.y < right_shoulder.y - 0.05:
        return "MANO LEVANTADA"

    left_open = left_wrist.x < left_shoulder.x - 0.18 and abs(left_wrist.y - left_shoulder.y) < 0.18
    right_open = right_wrist.x > right_shoulder.x + 0.18 and abs(right_wrist.y - right_shoulder.y) < 0.18
    if left_open and right_open:
        return "BRAZOS ABIERTOS"

    return None


def construir_respuesta(emocion, gesto):
    base = CONTENIDO_FILOSOFICO.get(emocion or "neutral", CONTENIDO_FILOSOFICO["neutral"]).copy()
    base["gesto"] = gesto
    base["mensaje_gesto"] = MENSAJES_GESTO.get(gesto, "")

    if gesto == "PULGAR ARRIBA":
        base["accion"] = "Vas bien. Conserva este impulso y conviertelo en una accion concreta hoy."
    elif gesto == "MANO LEVANTADA":
        base["pregunta"] = "Si tuvieras que formular una sola pregunta importante sobre tu estado actual, ¿cual seria?"
    elif gesto == "BRAZOS ABIERTOS":
        base["accion"] = "Estas mostrando apertura. Buen momento para escuchar una idea distinta sin defenderte de inmediato."

    return base


class FilosofVirtualProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self.state = {
            "emocion": None,
            "confianza": 0.0,
            "probabilidades": {},
            "gesto": None,
            "rostro_detectado": False,
            "respuesta": construir_respuesta(None, None),
        }

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        emotion = None
        probs = {}
        confidence = 0.0
        face_detected = False
        hand_gesture = None
        body_gesture = None

        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
        if len(faces) > 0 and MODELO is not None:
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            face_detected = True
            face_roi = gray[y : y + h, x : x + w]
            emotion, probs = predecir_emocion(face_roi)
            confidence = max(probs.values()) if probs else 0.0

            label = f"{EMOCIONES_ES.get(emotion, emotion)} {confidence:.0%}"
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 200, 120), 2)
            cv2.putText(output, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 120), 2)

        hands_result = self.hands.process(rgb)
        if hands_result.multi_hand_landmarks:
            for hand_landmarks in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detected = detectar_gesto_mano(hand_landmarks)
                if detected:
                    hand_gesture = detected
                    break

        pose_result = self.pose.process(rgb)
        if pose_result.pose_landmarks:
            body_gesture = detectar_gesto_pose(pose_result.pose_landmarks.landmark)

        gesture = hand_gesture or body_gesture
        response = construir_respuesta(emotion, gesture)

        overlay_lines = []
        if emotion:
            overlay_lines.append(f"Emocion: {EMOCIONES_ES.get(emotion, emotion)}")
        if gesture:
            overlay_lines.append(f"Gesto: {gesture}")

        for index, line in enumerate(overlay_lines):
            cv2.putText(
                output,
                line,
                (15, 30 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        with self.lock:
            self.state = {
                "emocion": emotion,
                "confianza": confidence,
                "probabilidades": probs,
                "gesto": gesture,
                "rostro_detectado": face_detected,
                "respuesta": response,
            }

        return av.VideoFrame.from_ndarray(output, format="bgr24")


def renderizar_panel_estado(state, mode):
    emocion = state.get("emocion")
    gesto = state.get("gesto")
    respuesta = state.get("respuesta", construir_respuesta(None, None))
    probs = state.get("probabilidades", {})

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if mode != "Solo gestos":
            label = EMOCIONES_ES.get(emocion, "Sin detectar")
            emoji = EMOJIS.get(emocion, "🧠")
            st.metric("Estado emocional", f"{emoji} {label}", f"{state.get('confianza', 0.0):.0%}")
        else:
            st.metric("Estado emocional", "Modo gestos", "")
    with col_b:
        st.metric("Gesto reconocido", gesto or "Sin gesto", "")
    with col_c:
        st.metric("Filosofo activo", respuesta.get("filosofo", "Kant"), "")

    st.markdown("### Reflexion en vivo")
    st.info(respuesta.get("frase", "Permite que el sistema observe tus expresiones para personalizar la reflexion."))
    st.write(f"**Pregunta:** {respuesta.get('pregunta', '¿Que quieres explorar hoy?')}")
    st.write(f"**Sugerencia:** {respuesta.get('accion', 'Mantente presente y observa como cambia tu expresion.')}")

    if respuesta.get("mensaje_gesto") and mode != "Solo emociones":
        st.success(respuesta["mensaje_gesto"])

    if probs and mode != "Solo gestos":
        ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)[:4]
        st.markdown("### Probabilidades mas altas")
        for emotion_key, value in ordered:
            st.write(f"- **{EMOCIONES_ES.get(emotion_key, emotion_key)}**: {value:.1%}")


def procesar_imagen_estatica(uploaded_file, detectar_gestos=True):
    img_pil = Image.open(uploaded_file).convert("RGB")
    image = np.array(img_pil)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    emotion = None
    probs = {}
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))
    if len(faces) > 0 and MODELO is not None:
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        emotion, probs = predecir_emocion(gray[y : y + h, x : x + w])

    gesture = None
    if detectar_gestos:
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            result = hands.process(image)
            if result.multi_hand_landmarks:
                gesture = detectar_gesto_mano(result.multi_hand_landmarks[0])

    state = {
        "emocion": emotion,
        "confianza": max(probs.values()) if probs else 0.0,
        "probabilidades": probs,
        "gesto": gesture,
        "rostro_detectado": len(faces) > 0,
        "respuesta": construir_respuesta(emotion, gesture),
    }
    return img_pil, state


st.title("📜 El Filosofo Virtual")
st.caption("Analiza tu expresion y tus gestos en tiempo real para ofrecerte una respuesta filosofica breve, clara y util.")

with st.sidebar:
    st.header("Configuracion")
    mode = st.radio(
        "Modo de analisis",
        ["Integral en vivo", "Solo emociones", "Solo gestos"],
        index=0,
    )
    st.markdown("---")
    if MODELO_PATH:
        st.success(f"Modelo cargado: {os.path.basename(MODELO_PATH)}")
    else:
        st.error("No se pudo cargar el modelo de emociones.")
        if ERRORES_MODELO:
            st.caption("Ultimo error detectado:")
            st.code(ERRORES_MODELO[-1])
    st.info(
        "La deteccion de gestos usa MediaPipe y no necesita dataset adicional. "
        "Para emociones se usa tu dataset FER local entrenado en este proyecto."
    )

tab_live, tab_backup = st.tabs(["Tiempo real", "Respaldo con imagen"])

with tab_live:
    st.subheader("Camara en vivo")
    st.write(
        "Permite el acceso a la camara. El sistema intentara detectar emociones faciales y gestos de manos o postura al mismo tiempo."
    )

    if REALTIME_AVAILABLE:
        ctx = webrtc_streamer(
            key="filosofo-virtual-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=FilosofVirtualProcessor,
            async_processing=True,
        )

        info_placeholder = st.empty()
        panel_placeholder = st.empty()

        if ctx.state.playing:
            info_placeholder.success("Analizando video en tiempo real...")
            while ctx.state.playing:
                if ctx.video_processor:
                    state = ctx.video_processor.get_state()
                    with panel_placeholder.container():
                        renderizar_panel_estado(state, mode)
                time.sleep(0.6)
        else:
            info_placeholder.info("Presiona Start para iniciar el analisis en vivo.")
    else:
        st.warning("El modo en vivo no esta disponible en este entorno.")
        st.code(REALTIME_IMPORT_ERROR or "Falta instalar streamlit-webrtc.")

with tab_backup:
    st.subheader("Analisis de respaldo")
    st.write("Si la camara en vivo falla en tu entorno, puedes probar con una imagen para validar que el modelo y la logica funcionan.")
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image, state = procesar_imagen_estatica(uploaded, detectar_gestos=(mode != "Solo emociones"))
        col1, col2 = st.columns([1, 1.1])
        with col1:
            st.image(image, caption="Entrada analizada", width=420)
        with col2:
            renderizar_panel_estado(state, mode)

st.markdown("---")
st.markdown(
    "Hecho con Streamlit, TensorFlow, OpenCV y MediaPipe para crear una experiencia "
    "academica de filosofia asistida por vision computacional."
)
