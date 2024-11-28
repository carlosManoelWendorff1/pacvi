import cv2
import torch
import warnings

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Contadores
count_inside = 0

# Rastreamento de pessoas
tracked_objects = {}

# Limite de proximidade para considerar uma pessoa como a mesma
PROXIMITY_THRESHOLD = 80  # Aumentando o threshold
FRAME_WIDTH = None  # Será definido após o primeiro frame

# Contador de ID único global
next_object_id = 0

def euclidean_distance(point1, point2):
    """Calcula a distância euclidiana entre dois pontos."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def track_and_count(detections, frame_width):
    warnings.filterwarnings("ignore", category=FutureWarning)
    """Rastreia e conta pessoas baseando-se no centro das caixas."""
    global count_inside, tracked_objects, next_object_id

    updated_tracked_objects = {}

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        center = (int((x1 + x2) // 2), int((y1 + y2) // 2))  # Convertido para inteiros

        # Verificar se o centro está próximo a algum rastreado anteriormente
        matched_id = None
        for obj_id, obj_data in tracked_objects.items():
            if euclidean_distance(center, obj_data["center"]) < PROXIMITY_THRESHOLD:
                matched_id = obj_id
                break

        if matched_id is None:
            # Novo objeto detectado com ID único
            updated_tracked_objects[next_object_id] = {"center": center, "exited": False, "entered": False}
            next_object_id += 1  # Incrementa o ID para o próximo objeto
        else:
            # Atualizar objeto existente
            previous_data = tracked_objects[matched_id]
            updated_tracked_objects[matched_id] = previous_data
            updated_tracked_objects[matched_id]["center"] = center

            # Verificar entrada (lado direito)
            if not previous_data["entered"] and center[0] > frame_width * 0.6:  # Verifica se está na extrema direita
                updated_tracked_objects[matched_id]["entered"] = True
                count_inside += 1
                print(f"ID {matched_id} entrou pela direita. Contagem aumentada: {count_inside}")

            # Verificar saída (lado esquerdo)
            if not previous_data["exited"] and center[0] < frame_width * 0.4:  # Verifica se está na extrema esquerda
                updated_tracked_objects[matched_id]["exited"] = True
                count_inside -= 1
                print(f"ID {matched_id} saiu pela esquerda. Contagem reduzida: {count_inside}")

    return updated_tracked_objects


# Inicializar captura de vídeo
video_path = "bus3.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if FRAME_WIDTH is None:
        FRAME_WIDTH = frame.shape[1]

    # Realizar inferência
    results = model(frame)

    # Filtrar apenas pessoas (classe 0 no YOLOv5)
    detections = results.pred[0].numpy()
    person_detections = [d for d in detections if int(d[5]) == 0]

    # Atualizar rastreamento e contadores
    tracked_objects = track_and_count(person_detections, FRAME_WIDTH)

    # Exibir contador no vídeo
    cv2.putText(frame, f"Pessoas dentro: {count_inside}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Desenhar caixas delimitadoras e centros
    for obj_id, obj_data in tracked_objects.items():
        center = obj_data["center"]
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {obj_id}", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Desenhar indicadores de entrada e saída
    for obj_id, obj_data in tracked_objects.items():
        center = obj_data["center"]
        if obj_data["entered"]:
            cv2.putText(frame, f"Entrou ID {obj_id}", (center[0] - 10, center[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if obj_data["exited"]:
            cv2.putText(frame, f"Saiu ID {obj_id}", (center[0] - 10, center[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar o frame processado
    cv2.imshow('Video', frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
