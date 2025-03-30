import cv2
import numpy as np
import mediapipe as mp
import torch
from scipy.spatial import distance as dis
import threading
import pyttsx3
import datetime
import time
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pathlib
# ----- Configurações e constantes globais -----
ID_DEVICE = '001'
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)  
COLOR_GREEN = (0, 255, 0)  

# Limiares para o método baseado em landmarks
min_frameEyes = 11
min_frameMouth = 35
min_toleranceEyes = 5.3
min_toleranceMouth = 1.4
last_capture_time = 0
capture_interval = 10  # Intervalo de 10 segundos entre os envios para o servidor

# Pontos de referência para os olhos e boca
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]
UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

# Inicialização dos contadores
frame_count = 0
frame_countM = 0
flagBocejo = 0

# ----- Inicialização dos modelos -----
# MediaPipe Face Mesh para o primeiro método
face_mesh = mp.solutions.face_mesh
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0,255,0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0,0,255), thickness=1, circle_radius=1)

face_model = face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# YOLOv5 para o segundo método
try:
    

    # Substitui temporariamente PosixPath por WindowsPath
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp3/weights/last.pt', force_reload=True)

    # Restaura PosixPath para evitar problemas futuros
    pathlib.PosixPath = temp
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    print("Continuando apenas com o método MediaPipe")
    yolo_model = None

# Inicialização do TTS
speech = pyttsx3.init()

# ----- Funções auxiliares para o método MediaPipe -----
def run_speech(speech, speech_message):
    """Função para reproduzir mensagem de aviso via TTS"""
    speech.say(speech_message)
    speech.runAndWait()

def draw_landmarks(image, outputs, land_mark, color):
    """Desenha os landmarks no rosto"""
    height, width = image.shape[:2]
             
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        point_scale = ((int)(point.x * width), (int)(point.y * height))
        cv2.circle(image, point_scale, 2, color, 1)

def euclidean_distance(image, top, bottom):
    """Calcula a distância euclidiana entre dois pontos"""
    height, width = image.shape[0:2]
            
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    
    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right):
    """Calcula a relação de aspecto (aspect ratio) entre pontos"""
    landmark = outputs.multi_face_landmarks[0]
            
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)
    
    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)
    
    aspect_ratio = left_right_dis / top_bottom_dis
    
    return aspect_ratio

def analyze_head_pose(face_landmarks, image):
    """Analisa a pose da cabeça e retorna o ângulo x (inclinação para baixo)"""
    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []
    
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm.x * img_w, lm.y * img_h)
                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * img_w
    cam_matrix = np.array([
        [focal_length, 0, img_h/2],
        [0, focal_length, img_w/2],
        [0, 0, 1]
    ])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
    
    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
    
    if not success:
        return 0
    
    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    x = angles[0] * 360  # Inclinação para baixo/cima
    y = angles[1] * 360  # Virar para esquerda/direita
    z = angles[2] * 360  # Inclinação lateral
    
    return x

def process_mediapipe(image):
    """Processa a imagem usando MediaPipe e retorna métricas de sonolência"""
    global frame_count, frame_countM, flagBocejo
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    outputs = face_model.process(image_rgb)
    image_rgb.flags.writeable = True
    
    # Valores padrão caso não haja face detectada
    ear = 0
    mar = 0
    head_angle = 0
    eyes_closed = 0
    yawning = 0
    head_drooping = 0
    
    if outputs.multi_face_landmarks:
        # Análise da pose da cabeça
        head_angle = analyze_head_pose(outputs.multi_face_landmarks[0], image)
        
        # Análise dos olhos
        ratio_left = get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
        ratio_right = get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
        ear = (ratio_left + ratio_right) / 2.0  # EAR - Eye Aspect Ratio
        
        # Análise da boca
        mar = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)  # MAR - Mouth Aspect Ratio
        
        # Lógica para determinar estado dos olhos
        if ear > min_toleranceEyes:
            frame_count += 1
            if frame_count > 2:
                eyes_closed = (frame_count / min_frameEyes) * 100
                eyes_closed = min(eyes_closed, 100)
        else:
            frame_count = 0
            
        # Lógica para determinar bocejo
        if mar < min_toleranceMouth:
            frame_countM += 1
            yawning = (frame_countM / min_frameMouth) * 100
            yawning = min(yawning, 100)
            flagBocejo = 1
        else:
            frame_countM = 0
            flagBocejo = 0
        
        # Lógica para determinar inclinação da cabeça
        if head_angle < -10:
            head_drooping = min(abs(head_angle / 20) * 100, 100)  # Normalizado até -20 graus sendo 100%
    
    return {
        'ear': ear,
        'mar': mar,
        'head_angle': head_angle,
        'eyes_closed_percent': eyes_closed,
        'yawning_percent': yawning,
        'head_drooping_percent': head_drooping
    }

def process_yolo(image):
    """Processa a imagem usando o modelo YOLO e retorna as probabilidades de cada classe"""
    if yolo_model is None:
        return {'awake': 100, 'sleeping': 0, 'yawning': 0}
    
    results = yolo_model(image)
    
    # Valores padrão caso não haja detecção
    class_probs = {'awake': 100, 'sleeping': 0, 'yawning': 0}
    
    # Extrai as classes e probabilidades das detecções
    if len(results.xyxy[0]) > 0:
        # Organiza as detecções por confiança (do maior para o menor)
        detections = results.xyxy[0].cpu().numpy()
        
        # Para cada detecção, obtém a classe e a confiança
        for detection in detections:
            confidence = detection[4] * 100  # Converte para porcentagem
            class_idx = int(detection[5])
            
            # Mapeia o índice da classe para o nome (ajuste conforme seu modelo)
            class_names = {1: 'awake', 0: 'sleeping', 2: 'yawning'}
            if class_idx in class_names:
                class_name = class_names[class_idx]
                class_probs[class_name] = confidence


    
    return class_probs

# ----- Sistema de Lógica Fuzzy -----
def setup_fuzzy_system():
    """Configura o sistema de lógica fuzzy"""
    # Variáveis de entrada
    mediapipe_eyes = ctrl.Antecedent(np.arange(0, 101, 1), 'mediapipe_eyes')
    mediapipe_yawn = ctrl.Antecedent(np.arange(0, 101, 1), 'mediapipe_yawn')
    mediapipe_head = ctrl.Antecedent(np.arange(0, 101, 1), 'mediapipe_head')
    yolo_sleeping = ctrl.Antecedent(np.arange(0, 101, 1), 'yolo_sleeping')
    yolo_yawning = ctrl.Antecedent(np.arange(0, 101, 1), 'yolo_yawning')
    
    # Variável de saída
    drowsiness = ctrl.Consequent(np.arange(0, 101, 1), 'drowsiness')
    
    # Funções de pertinência para as entradas do MediaPipe
    mediapipe_eyes['low'] = fuzz.trimf(mediapipe_eyes.universe, [0, 0, 30])
    mediapipe_eyes['medium'] = fuzz.trimf(mediapipe_eyes.universe, [20, 50, 80])
    mediapipe_eyes['high'] = fuzz.trimf(mediapipe_eyes.universe, [70, 100, 100])
    
    mediapipe_yawn['low'] = fuzz.trimf(mediapipe_yawn.universe, [0, 0, 30])
    mediapipe_yawn['medium'] = fuzz.trimf(mediapipe_yawn.universe, [20, 50, 80])
    mediapipe_yawn['high'] = fuzz.trimf(mediapipe_yawn.universe, [70, 100, 100])
    
    mediapipe_head['low'] = fuzz.trimf(mediapipe_head.universe, [0, 0, 30])
    mediapipe_head['medium'] = fuzz.trimf(mediapipe_head.universe, [20, 50, 80])
    mediapipe_head['high'] = fuzz.trimf(mediapipe_head.universe, [70, 100, 100])
    
    # Funções de pertinência para as entradas do YOLO
    yolo_sleeping['low'] = fuzz.trimf(yolo_sleeping.universe, [0, 0, 30])
    yolo_sleeping['medium'] = fuzz.trimf(yolo_sleeping.universe, [20, 50, 80])
    yolo_sleeping['high'] = fuzz.trimf(yolo_sleeping.universe, [70, 100, 100])
    
    yolo_yawning['low'] = fuzz.trimf(yolo_yawning.universe, [0, 0, 30])
    yolo_yawning['medium'] = fuzz.trimf(yolo_yawning.universe, [20, 50, 80])
    yolo_yawning['high'] = fuzz.trimf(yolo_yawning.universe, [70, 100, 100])
    
    # Funções de pertinência para a saída
    drowsiness['alert'] = fuzz.trimf(drowsiness.universe, [0, 0, 30])
    drowsiness['tired'] = fuzz.trimf(drowsiness.universe, [20, 50, 80])
    drowsiness['sleepy'] = fuzz.trimf(drowsiness.universe, [70, 100, 100])
    
    # Regras de lógica fuzzy
    rule1 = ctrl.Rule(mediapipe_eyes['high'] & yolo_sleeping['high'], drowsiness['sleepy'])
    rule2 = ctrl.Rule(mediapipe_eyes['high'] & mediapipe_head['high'], drowsiness['sleepy'])
    rule3 = ctrl.Rule(mediapipe_yawn['high'] & yolo_yawning['high'], drowsiness['tired'])
    rule4 = ctrl.Rule(mediapipe_eyes['medium'] & yolo_sleeping['medium'], drowsiness['tired'])
    rule5 = ctrl.Rule(mediapipe_eyes['low'] & yolo_sleeping['low'], drowsiness['alert'])
    rule6 = ctrl.Rule(mediapipe_eyes['high'] | yolo_sleeping['high'], drowsiness['sleepy'])
    rule7 = ctrl.Rule(mediapipe_yawn['high'] | yolo_yawning['high'], drowsiness['tired'])
    rule8 = ctrl.Rule(mediapipe_head['high'] & (mediapipe_eyes['medium'] | yolo_sleeping['medium']), drowsiness['sleepy'])
    rule9 = ctrl.Rule(mediapipe_eyes['medium'] & mediapipe_yawn['medium'], drowsiness['tired'])
    rule10 = ctrl.Rule(yolo_sleeping['medium'] & yolo_yawning['medium'], drowsiness['tired'])
    
    # Sistema de controle
    drowsiness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
    drowsiness_simulator = ctrl.ControlSystemSimulation(drowsiness_ctrl)
    
    return drowsiness_simulator

def evaluate_drowsiness(simulator, mediapipe_results, yolo_results):
    """Avalia o nível de sonolência usando o sistema fuzzy"""
    # Define as entradas para o simulador
    simulator.input['mediapipe_eyes'] = mediapipe_results['eyes_closed_percent']
    simulator.input['mediapipe_yawn'] = mediapipe_results['yawning_percent']
    simulator.input['mediapipe_head'] = mediapipe_results['head_drooping_percent']
    simulator.input['yolo_sleeping'] = yolo_results['sleeping']
    simulator.input['yolo_yawning'] = yolo_results['yawning']
    
    # Calcula a saída
    try:
        simulator.compute()
        drowsiness_level = simulator.output['drowsiness']
    except:
        # Em caso de erro no sistema fuzzy, usamos uma heurística simples
        drowsiness_level = max(
            mediapipe_results['eyes_closed_percent'],
            yolo_results['sleeping'],
            (mediapipe_results['yawning_percent'] + yolo_results['yawning']) / 2
        )
    
    # Classificação do estado
    if drowsiness_level >= 70:
        state = "DORMINDO"
        color = COLOR_RED
    elif drowsiness_level >= 30:
        state = "CANSADO"
        color = (0, 165, 255)  # Laranja
    else:
        state = "ALERTA"
        color = COLOR_GREEN
    
    return {
        'level': drowsiness_level,
        'state': state,
        'color': color
    }

def visualize_results(image, mediapipe_results, yolo_results, drowsiness_result):
    """Visualiza os resultados na imagem"""
    # Informações do MediaPipe
    cv2.putText(image, f"EAR: {mediapipe_results['ear']:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.putText(image, f"MAR: {mediapipe_results['mar']:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    cv2.putText(image, f"Cabeça: {mediapipe_results['head_angle']:.1f}º", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    # Informações do YOLO
    if yolo_model is not None:
        cv2.putText(image, f"YOLO Dormindo: {yolo_results['sleeping']:.1f}%", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
        cv2.putText(image, f"YOLO Bocejando: {yolo_results['yawning']:.1f}%", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    # Resultado final do sistema fuzzy
    cv2.putText(image, f"Estado: {drowsiness_result['state']}", (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, drowsiness_result['color'], 2)
    cv2.putText(image, f"Nível: {drowsiness_result['level']:.1f}%", (10, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, drowsiness_result['color'], 2)
    
    # Barra de nível de sonolência
    bar_length = 200
    filled_length = int(bar_length * drowsiness_result['level'] / 100)
    
    # Desenha barra de progresso
    cv2.rectangle(image, (10, 240), (10 + bar_length, 260), COLOR_WHITE, 2)
    cv2.rectangle(image, (10, 240), (10 + filled_length, 260), drowsiness_result['color'], -1)
    
    return image

def main_video():
    """Executa a detecção de sonolência usando vídeo ao vivo."""
    fuzzy_simulator = setup_fuzzy_system()
    cap = cv2.VideoCapture(2)  # Use 0 para webcam padrão
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return
    
    last_alert_time = 0
    alert_cooldown = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame da câmera.")
            break
        
        mediapipe_results = process_mediapipe(frame)
        yolo_results = process_yolo(frame)
        drowsiness_result = evaluate_drowsiness(fuzzy_simulator, mediapipe_results, yolo_results)
        frame = visualize_results(frame, mediapipe_results, yolo_results, drowsiness_result)
        
        current_time = time.time()
        if drowsiness_result['level'] >= 70 and (current_time - last_alert_time) > alert_cooldown:
            threading.Thread(target=run_speech, args=(speech, "Alerta de sonolência! Pare o veículo.")).start()
            last_alert_time = current_time
        elif drowsiness_result['level'] >= 40 and (current_time - last_alert_time) > alert_cooldown:
            threading.Thread(target=run_speech, args=(speech, "Você parece cansado. Considere fazer uma pausa.")).start()
            last_alert_time = current_time
        
        cv2.imshow("Fuzzy Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main_image(image_path):
    """Executa a detecção de sonolência em uma imagem estática."""
    fuzzy_simulator = setup_fuzzy_system()
    frame = cv2.imread(image_path)
    if frame is None:
        print("Erro: Não foi possível carregar a imagem.")
        return
    
    mediapipe_results = process_mediapipe(frame)
    yolo_results = process_yolo(frame)
    drowsiness_result = evaluate_drowsiness(fuzzy_simulator, mediapipe_results, yolo_results)
    frame = visualize_results(frame, mediapipe_results, yolo_results, drowsiness_result)
    
    cv2.imshow("Fuzzy Drowsiness Detection - Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODE = "video"  # Altere para "image" para processar uma imagem específica
    IMAGE_PATH = "img_test.jpg"  # Defina o caminho da imagem caso use o modo "image"
    
    if MODE == "video":
        main_video()
    elif MODE == "image":
        main_image(IMAGE_PATH)
    else:
        print("Modo inválido. Escolha 'video' ou 'image'.")

