import cv2
import mediapipe as mp
from deepface import DeepFace
from tqdm import tqdm
import warnings
import os
from collections import Counter

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)

def detect_emotions(input_video, output_video, summary_output_path):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    # open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    emotion_counter = Counter()

    for frame in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # frame conversion to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # face detection
        results = face_detection.process(rgb_frame)

        frame_emotions = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                padding = 10
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = min(w + 2 * padding, iw - x)
                h = min(h + 2 * padding, ih - y)

                # extract face region
                face_region = frame[y:y+h, x:x+w]

                try:
                    # emotion analysis
                    face_region_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
                    emotion_analysis = DeepFace.analyze(face_region_rgb, actions=['emotion'], enforce_detection=False)

                    # verify if the result is a list
                    if isinstance(emotion_analysis, list):
                        emotion_analysis = emotion_analysis[0]

                    # extract dominant emotion and confidence
                    dominant_emotion = emotion_analysis['dominant_emotion']
                    confidence = emotion_analysis['emotion'][dominant_emotion]

                    if confidence > 0.7:  # only consider emotions with confidence > 0.7
                        frame_emotions.append({
                            "dominant_emotion": dominant_emotion,
                            "confidence": confidence,
                            "bounding_box": [x, y, w, h]
                        })

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                        emotion_counter[dominant_emotion] += 1

                except Exception as e:
                    print(f"Erro na análise de emoções: {e}")

        # write frame to output video
        out.write(frame)

    # release resources
    cap.release()
    out.release()

    with open(summary_output_path, 'w') as f:
        f.write(f"Identificao Emocoes - Total de frames analisados: {total_frames}\n")
        f.write(f"Identificao Emocoes - Total de tipos de emocoes detectadas: {len(emotion_counter)}\n")
        for emotion, count in emotion_counter.items():
            f.write(f"Identificao Emocoes - {emotion}: {count}\n")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_video = os.path.join(script_dir, 'videos/video.mp4')
    output_video = os.path.join(script_dir, 'videos/emotion_recognition_output.mp4')
    output_summary = os.path.join(script_dir, 'summaries/emotion_summary.txt')

    detect_emotions(input_video, output_video, output_summary)
