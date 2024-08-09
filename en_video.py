import cv2
import os
from ultralytics import YOLOv10
import supervision as sv

#pip install -q git+https://github.com/THU-MIG/yolov10.git

# Cargar el modelo YOLO
model_path = "/path/to/model"

model = YOLOv10(model_path)

# Diccionario de nombres de clases
class_names = ['10B', '10C', '10E', '10O', '11B', '11C', '11E', '11O', '12B', '12C', '12E', '12O',
               '1B', '1C', '1E', '1O', '2B', '2C', '2E', '2O', '3B', '3C', '3E', '3O', '4B', '4C',
               '4E', '4O', '5B', '5C', '5E', '5O', '6B', '6C', '6E', '6O', '7B', '7C', '7E', '7O',
               '8B', '8C', '8E', '8O', '9B', '9C', '9E', '9O', 'J']

class_dict = {i: name for i, name in enumerate(class_names)}

def clasificar_cartas(array_detecciones):
    cartas = {"E": [], "C": [], "B": [], "O": [], "J": []}
    for deteccion in array_detecciones:
        if len(deteccion) > 1:
            try:
                numero = int(deteccion[:-1])  # Todo menos el último carácter representa el número
                palo = deteccion[-1]  # El último carácter representa el palo
                if palo in cartas:
                    cartas[palo].append(numero)
            except ValueError:
                print(f"Error al convertir {deteccion} a número y palo")
        else:
            palo = deteccion
            numero = True

    return cartas

# Función para calcular los puntos
def calcular_puntos(cartas):
    puntos = 0
    total_elementos = sum(len(lista) for lista in cartas.values() if lista)
    if total_elementos != 3:
        return "Solo puede haber 3 cartas para el truco"
    for palos in cartas:
        for num in cartas[palos]:
            if num in [8, 9]:
                return "Error: Carta no apta para el truco"

        if len(cartas[palos]) > 1:
            puntos = 20
            cartas_suma = [num for num in cartas[palos] if num not in [10, 11, 12]]
            cartas_no_suma = [num for num in cartas[palos] if num in [10, 11, 12]]

            cartas_no_suma = [0] * len(cartas_no_suma)

            cartas_validas = sorted(cartas_suma + cartas_no_suma, reverse=True)

            puntos += sum(cartas_validas[:2])

    return puntos

input_video_path = '/path/to-video.mp4'
output_video_path = '/content/video_procesado.mp4'

cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_counter = 0
last_puntos_text = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for bbox, conf, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, bbox)
        label = class_dict[int(cls)]
        confidence = f"Conf: {conf:.2f}"

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        frame = cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    detections = sv.Detections.from_ultralytics(results)
    array_detecciones = detections.__getitem__('class_name')

    cartas = clasificar_cartas(array_detecciones)
    puntos = calcular_puntos(cartas)

    if frame_counter % 8 == 0:
        last_puntos_text = f"Puntos: {puntos}"
    
    if last_puntos_text:
        frame = cv2.putText(frame, last_puntos_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    out.write(frame)

    frame_counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()
