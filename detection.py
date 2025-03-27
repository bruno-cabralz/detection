import boto3
import cv2
import json
import numpy as np
from mmdet.apis import init_detector, inference_detector
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple

# Constantes do MMDetection
CONFIG_FILE = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
CHECKPOINT_FILE = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
DEVICE = 'cuda:0'
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

# Inicialização do DeepSORT e AWS Rekognition
tracker = DeepSort(max_age=30)
rekognition_client = boto3.client('rekognition', region_name='us-east-1')

# Configurações de rastreamento e contagem
LINE_Y_POSITION = 400
jeans_zone_bbox = (100, 100, 400, 400)
count_in, count_out_with_bag, count_interested = 0, 0, 0
tracked_ids_in, tracked_ids_out, tracked_ids_jeans = set(), set(), set()


def download_video_from_s3(s3_client: boto3.client, bucket: str, filename: str):
    """Baixa o vídeo do S3."""
    s3_client.download_file(bucket, filename, filename)


def analyze_face(image_bytes: bytes) -> dict:
    """Analisa o rosto extraído usando Amazon Rekognition."""
    response = rekognition_client.detect_faces(
        Image={'Bytes': image_bytes}, Attributes=['ALL']
    )
    if response['FaceDetails']:
        face_details = response['FaceDetails'][0]
        return {
            'age_range': face_details['AgeRange'],
            'gender': face_details['Gender']['Value'],
            'emotions': [
                {'type': emotion['Type'], 'confidence': emotion['Confidence']}
                for emotion in face_details['Emotions']
            ],
        }
    return None


def bbox_inside_jeans_zone(bbox: Tuple[int, int, int, int]) -> bool:
    """Verifica se a bounding box está dentro da área de roupas jeans."""
    x, y, w, h = bbox
    return (jeans_zone_bbox[0] <= x <= jeans_zone_bbox[2] and
            jeans_zone_bbox[1] <= y <= jeans_zone_bbox[3])


def extract_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bytes:
    """Extrai e redimensiona o rosto com base na bounding box."""
    x, y, w, h = bbox
    face = frame[y:y+h, x:x+w]
    _, buf = cv2.imencode('.jpg', face)
    return buf.tobytes()


def process_video(video_path: str, output_path: str, is_jeans_area: bool = False) -> List[dict]:
    """Processa o vídeo, salva o vídeo processado e retorna os dados JSON."""
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    json_data = []

    global count_in, count_out_with_bag, count_interested

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecção de pessoas
        results = inference_detector(model, frame)
        person_bboxes = [bbox[:4] for bbox in results[0][0] if bbox[4] > 0.5]
        tracked_objects = tracker.update_tracks(person_bboxes, frame=frame)

        for obj in tracked_objects:
            obj_id = obj.track_id
            x, y, w, h = map(int, obj.to_ltwh())
            center_y = y + h // 2

            if is_jeans_area:
                if bbox_inside_jeans_zone((x, y, w, h)) and obj_id not in tracked_ids_jeans:
                    tracked_ids_jeans.add(obj_id)
                    face_analysis = analyze_face(extract_face(frame, (x, y, w, h)))
                    json_data.append({
                        'interaction_type': 'jeans_interest',
                        'person_id': obj_id,
                        'age_range': face_analysis['age_range'],
                        'gender': face_analysis['gender'],
                        'emotions': face_analysis['emotions'],
                    })
            else:
                # Contagem de entrada e saída
                if obj_id not in tracked_ids_in and center_y < LINE_Y_POSITION:
                    tracked_ids_in.add(obj_id)
                    count_in += 1
                elif obj_id not in tracked_ids_out and center_y > LINE_Y_POSITION and obj['classes'] == ['bag']:
                    tracked_ids_out.add(obj_id)
                    count_out_with_bag += 1

            # Desenho das informações na frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Exibe contador no vídeo
        cv2.putText(frame, f"Entrada: {count_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Saída com Sacola: {count_out_with_bag}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    return json_data


def save_json(data: List[dict], filename: str):
    """Salva os dados processados em um arquivo JSON."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    """Função principal para processar vídeos."""
    bucket_name = 'meu-bucket-s3'
    video_files = ['entrada.mp4', 'jeans.mp4']
    final_data = []

    for video_file in video_files:
        download_video_from_s3(s3_client=boto3.client('s3'), bucket=bucket_name, filename=video_file)
        is_jeans_area = 'jeans' in video_file
        video_output = f'processed_{video_file}'
        data = process_video(video_file, video_output, is_jeans_area)
        final_data.extend(data)

    save_json(final_data, 'output_data.json')


if __name__ == "__main__":
    main()
