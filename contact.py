import boto3
import cv2
import numpy as np
import csv
from datetime import datetime
import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

# Configurações AWS
s3_client = boto3.client('s3', region_name='us-east-1')
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
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0

# Função para processar vídeo e contar pessoas que estão entrando e saindo da área
def process_entrance_video(video_path, polygon_points, output_video_path, detector):
    cap = cv2.VideoCapture(video_path)
    entrance_count = 0
    exit_count = 0
    frame_count = 0
    counted_ids = set()
    exited_ids = set()
    demographics = {}
    exit_demographics = {}

    # Configurar gravação de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detecção de pessoas usando MMDetection
        result = inference_detector(detector, frame)
        bboxes = result[0][0]  # Assumindo que a classe 'person' é a primeira classe

        # Filtrar bounding boxes com confiança maior que 0.8
        rects = [bbox[:4].astype(int) for bbox in bboxes if bbox[4] > 0.8]

        for bbox in rects:
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            if point_in_polygon(centroid, polygon_points):
                if bbox[3] - bbox[1] > 0:  # Verifica se a pessoa está se movendo para cima (entrando)
                    entrance_count += 1
                    counted_ids.add(tuple(bbox))

                    # Chamada ao Rekognition para reconhecimento de gênero, idade, sorriso e emoções
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    face_response = rekognition_client.detect_faces(
                        Image={'Bytes': frame_bytes},
                        Attributes=['ALL']
                    )

                    for face_detail in face_response['FaceDetails']:
                        age_range = face_detail['AgeRange']
                        gender = face_detail['Gender']['Value']
                        smile = face_detail['Smile']['Value']
                        emotions = face_detail['Emotions']
                        primary_emotion = max(emotions, key=lambda x: x['Confidence'])['Type']
                        demographics[tuple(bbox)] = {
                            'age_range': age_range,
                            'gender': gender,
                            'smile': smile,
                            'emotion': primary_emotion
                        }

                elif bbox[3] - bbox[1] < 0:  # Verifica se a pessoa está se movendo para baixo (saindo)
                    exit_count += 1
                    exited_ids.add(tuple(bbox))

                    # Chamada ao Rekognition para reconhecimento de gênero, idade, sorriso e emoções
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    face_response = rekognition_client.detect_faces(
                        Image={'Bytes': frame_bytes},
                        Attributes=['ALL']
                    )

                    for face_detail in face_response['FaceDetails']:
                        age_range = face_detail['AgeRange']
                        gender = face_detail['Gender']['Value']
                        smile = face_detail['Smile']['Value']
                        emotions = face_detail['Emotions']
                        primary_emotion = max(emotions, key=lambda x: x['Confidence'])['Type']
                        exit_demographics[tuple(bbox)] = {
                            'age_range': age_range,
                            'gender': gender,
                            'smile': smile,
                            'emotion': primary_emotion
                        }

        # Desenhar polígonos e contadores no frame
        cv2.polylines(frame, [np.array(polygon_points, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, f'Entradas: {entrance_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Saidas: {exit_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for bbox in rects:
            centroid = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            text = f'ID {tuple(bbox)}'
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (0, 255, 0), -1)
            if tuple(bbox) in demographics:
                demo = demographics[tuple(bbox)]
                demo_text = f'{demo["age_range"]["Low"]}-{demo["age_range"]["High"]} {demo["gender"]} {demo["smile"]} {demo["emotion"]}'
                cv2.putText(frame, demo_text, (centroid[0] - 10, centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Desenhar bounding boxes no frame
        for bbox in rects:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Gravar frame processado
        out.write(frame)

    cap.release()
    out.release()
    return entrance_count, exit_count, demographics, exit_demographics

# Função principal
def main():
    bucket_name = 'poc-cea'
    videos = {
        'entrada': 'MHDX_ch4_main_20250309180000_20250309183000.mp4',
    }

    # Defina manualmente os pontos do polígono aqui
    entrance_polygon_points = [(227, 924), (773, 365), (786, 372), (237, 932)]

    # Inicializar modelo de detecção
    detector_config = '/home/ubuntu/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    detector_checkpoint = '/home/ubuntu/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

    detector = init_detector(detector_config, detector_checkpoint, device='cuda:0')

    results = {}

    for area, video_key in videos.items():
        # Extrair data e hora do nome do vídeo
        video_name = os.path.basename(video_key)
        date_str = video_name.split('_')[3]
        time_str = date_str[8:14]
        date = datetime.strptime(date_str[0:8], '%Y%m%d')
        time = datetime.strptime(time_str, '%H%M%S')

        # Criar diretórios baseados na data e hora
        date_folder = date.strftime('%d%m%Y')
        time_folder = time.strftime('%H')
        download_path = f'/home/ubuntu/videos/{area}/{date_folder}/{time_folder}/{video_name}'
        output_video_path = f'/home/ubuntu/videos/{area}/{date_folder}/{time_folder}/processed_{video_name}'

        # Baixar vídeo do S3
        download_video(bucket_name, video_key, download_path)

        if area == 'entrada':
            entrance_count, exit_count, demographics, exit_demographics = process_entrance_video(download_path, entrance_polygon_points, output_video_path, detector)
            results[area] = {
                'entrance_count': entrance_count,
                'exit_count': exit_count,
                'demographics': demographics,
                'exit_demographics': exit_demographics
            }

    # Salvar resultados em um arquivo CSV
    with open('/home/ubuntu/videos/results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Área', 'Entradas', 'Saídas', 'Demografia', 'Demografia Saída'])

        for area, data in results.items():
            if area == 'entrada':
                demographics_str = '; '.join([f'{d["age_range"]["Low"]}-{d["age_range"]["High"]} {d["gender"]} {d["smile"]} {d["emotion"]}' for d in data['demographics'].values()])
                exit_demographics_str = '; '.join([f'{d["age_range"]["Low"]}-{d["age_range"]["High"]} {d["gender"]} {d["smile"]} {d["emotion"]}' for d in data['exit_demographics'].values()])
                writer.writerow([area, data['entrance_count'], data['exit_count'], demographics_str, exit_demographics_str])

    # Exibir resultados
    for area, data in results.items():
        print(f'Área: {area}')
        if area == 'entrada':
            print(f'Entradas: {data["entrance_count"]}')
            print(f'Saídas: {data["exit_count"]}')
        print(f'Demografia Entrada: {data["demographics"]}')
        if area == 'entrada':
            print(f'Demografia Saída: {data["exit_demographics"]}')
        print('---')

if __name__ == '__main__':
    main()