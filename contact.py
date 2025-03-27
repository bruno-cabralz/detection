import boto3
import cv2
import numpy as np
import json
import os
from collections import defaultdict
from datetime import datetime
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

# Configurações AWS
rekognition_client = boto3.client('rekognition', region_name='us-east-1')

# Função para baixar vídeo do S3
def download_video(bucket_name, video_key, download_path):
    if not os.path.exists(download_path):
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        s3_client.download_file(bucket_name, video_key, download_path)
        print(f"Downloaded {video_key} to {download_path}")
    else:
        print(f"File {download_path} already exists. Skipping download.")

# Função para verificar se um ponto está dentro do polígono
def point_in_polygon(point, polygon):
    point = (float(point[0]), float(point[1]))
    polygon = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Função para enviar o melhor frame da face para o Rekognition
def process_face_with_rekognition(frame, bounding_box):
    x1, y1, x2, y2 = bounding_box
    face_frame = frame[y1:y2, x1:x2]
    _, buffer = cv2.imencode('.jpg', face_frame)
    frame_bytes = buffer.tobytes()

    response = rekognition_client.detect_faces(
        Image={'Bytes': frame_bytes},
        Attributes=['ALL']
    )

    if response['FaceDetails']:
        face_detail = response['FaceDetails'][0]
        age_range = face_detail['AgeRange']
        gender = face_detail['Gender']['Value']
        emotions = face_detail['Emotions']
        primary_emotion = max(emotions, key=lambda x: x['Confidence'])['Type']
        return {'age_range': age_range, 'gender': gender, 'emotion': primary_emotion}
    return {}

# Função para processar vídeo e contar pessoas na entrada e saída
def process_entrance_video(video_path, polygon_points, detector, min_confidence=0.8):
    cap = cv2.VideoCapture(video_path)
    entrance_count = 0
    exit_count = 0
    counted_ids = set()
    demographics = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = inference_detector(detector, frame)
        bboxes = result[0][0]  # Assumindo que a classe 'person' é a primeira classe
        rects = [bbox[:4].astype(int) for bbox in bboxes if bbox[4] > min_confidence]

        for bbox in rects:
            object_id = str(bbox)  # Usar bounding box como identificador único
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            if object_id not in counted_ids:
                if point_in_polygon(centroid, polygon_points):
                    entrance_count += 1
                    counted_ids.add(object_id)

                    face_data = process_face_with_rekognition(frame, bbox)
                    if face_data:
                        demographics[object_id] = face_data

        # Gravar frame processado (opcional)
        # out.write(frame)

    cap.release()
    return entrance_count, demographics

# Função para processar vídeo na área de roupas
def process_clothing_area_video(video_path, polygon_points, detector, min_time_in_area=5, min_confidence=0.8):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    people_in_area = defaultdict(int)
    demographics = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = inference_detector(detector, frame)
        bboxes = result[0][0]
        rects = [bbox[:4].astype(int) for bbox in bboxes if bbox[4] > min_confidence]

        for bbox in rects:
            object_id = str(bbox)  # Usar bounding box como identificador único
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

            if point_in_polygon(centroid, polygon_points):
                people_in_area[object_id] += 1

                if people_in_area[object_id] >= min_time_in_area:
                    face_data = process_face_with_rekognition(frame, bbox)
                    if face_data:
                        demographics[object_id] = face_data

        frame_count += 1

    cap.release()
    return len(demographics), demographics

# Função principal
def main():
    bucket_name = 'poc-cea'
    videos = {
        'entrada': 'MHDX_ch4_main_20250309180000_20250309183000.mp4',
        'area_roupas_1': 'MHDX_ch5_main_20250309190004_20250309193004.mp4',
    }

    # Defina manualmente os pontos do polígono
    entrance_polygon_points = [(227, 924), (773, 365), (786, 372), (237, 932)]
    clothing_area_polygon_points = [(110, 197), (66, 1008), (954, 733), (941, 250), (455,30)]

    # Inicializar o modelo de detecção
    detector_config = '/home/ubuntu/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    detector_checkpoint = '/home/ubuntu/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
    detector = init_detector(detector_config, detector_checkpoint, device='cuda:0')

    results = {}

    for area, video_key in videos.items():
        video_path = f"downloads/{video_key}"  # Ajuste o caminho conforme necessário
        if area == 'entrada':
            entrance_count, entrance_demographics = process_entrance_video(video_path, entrance_polygon_points, detector)
            results[area] = {'entrance_count': entrance_count, 'demographics': entrance_demographics}
        elif area == 'area_roupas_1':
            clothing_count, clothing_demographics = process_clothing_area_video(video_path, clothing_area_polygon_points, detector)
            results[area] = {'clothing_count': clothing_count, 'demographics': clothing_demographics}

    # Salvar resultados em formato JSON
    output_file = f"results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em {output_file}")

if __name__ == "__main__":
    main()
