import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

def is_hand_near_face(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    mouth_left = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value]
    mouth_right = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]

    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array([point1.x, point1.y]) - np.array([point2.x, point2.y]))

    left_hand_on_face = (calculate_distance(left_wrist, nose) < 0.1 or
                         calculate_distance(left_wrist, left_eye) < 0.1 or
                         calculate_distance(left_wrist, right_eye) < 0.1 or
                         calculate_distance(left_wrist, mouth_left) < 0.1 or
                         calculate_distance(left_wrist, mouth_right) < 0.1)

    right_hand_on_face = (calculate_distance(right_wrist, nose) < 0.1 or
                          calculate_distance(right_wrist, left_eye) < 0.1 or
                          calculate_distance(right_wrist, right_eye) < 0.1 or
                          calculate_distance(right_wrist, mouth_left) < 0.1 or
                          calculate_distance(right_wrist, mouth_right) < 0.1)

    return left_hand_on_face or right_hand_on_face

def are_both_arms_raised(landmarks):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y

    return left_arm_up and right_arm_up

def is_left_arm_raised(landmarks):
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]

    left_arm_up = left_elbow.y < left_eye.y

    return left_arm_up

def is_right_arm_raised(landmarks):
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    right_arm_up = right_elbow.y < right_eye.y

    return right_arm_up

def is_waving(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    return (left_wrist.y < left_elbow.y) or (right_wrist.y < right_elbow.y)

def is_handshaking(landmarks):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return (left_wrist.x < left_shoulder.x) and (right_wrist.x > right_shoulder.x)

def is_nodding(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    return nose.y < ((left_shoulder.y + right_shoulder.y) / 2)

def recognize_activities(recognizer):
    input_video = 'videos/video.mp4'
    output_video = 'videos/activity_recognition_output.mp4'

    cap = cv2.VideoCapture(input_video)
    latest_gesture = None

    if not cap.isOpened():
        print("Erro ao abrir o video.")
        return

    # obtain video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    activities = []
    both_arms_raised_count = 0
    both_arms_raised = False
    left_arm_raised_count = 0
    left_arm_raised = False
    right_arm_raised_count = 0
    right_arm_raised = False
    wave_count = 0
    wave = False
    handshake_count = 0
    handshake = False
    nod_count = 0
    nod = False
    hand_near_face_count = 0
    hand_near_face = False

    for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # If a gesture is detected, display it on the frame
        cv2.putText(frame, f"Ultimo Gesto Detectado: {latest_gesture}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 140, 255), 2, cv2.LINE_AA)

        original_frame = frame.copy()

        timestamp = int((frame_idx / fps) * 1000)
        if (timestamp == 0):
            continue

        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # gesture recognition
        recognition_result = recognizer.recognize_for_video(rgb_frame, timestamp)

        if recognition_result.gestures:
            for gesture in recognition_result.gestures:
                for i in range(len(gesture)):
                    top_gesture = gesture[i].category_name
                    if (top_gesture == "None"):
                        continue
                    latest_gesture = top_gesture

                    activities.append(gesture[i].category_name)
                    cv2.putText(frame, f"Gesto: {top_gesture}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255),
                                2, cv2.LINE_AA)

        results = pose.process(original_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if are_both_arms_raised(results.pose_landmarks.landmark):
                if not both_arms_raised:
                    both_arms_raised = True
                    both_arms_raised_count += 1
                left_arm_raised = False
                right_arm_raised = False
            elif is_right_arm_raised(results.pose_landmarks.landmark):
                if not right_arm_raised:
                    right_arm_raised = True
                    right_arm_raised_count += 1
                both_arms_raised = False
                left_arm_raised = False
            elif is_left_arm_raised(results.pose_landmarks.landmark):
                if not left_arm_raised:
                    left_arm_raised = True
                    left_arm_raised_count += 1
                both_arms_raised = False
                right_arm_raised = False
            else:
                both_arms_raised = False
                left_arm_raised = False
                right_arm_raised = False

            if is_waving(results.pose_landmarks.landmark):
                if not wave:
                    wave = True
                    wave_count += 1
            else:
                wave = False

            if is_handshaking(results.pose_landmarks.landmark):
                if not handshake:
                    handshake = True
                    handshake_count += 1
            else:
                handshake = False

            if is_nodding(results.pose_landmarks.landmark):
                if not nod:
                    nod = True
                    nod_count += 1
            else:
                nod = False

            if is_hand_near_face(results.pose_landmarks.landmark):
                if not hand_near_face:
                    hand_near_face = True
                    hand_near_face_count += 1
            else:
                hand_near_face = False

            # Show statistics on the frame
            cv2.putText(frame, f'Ambos os Bracos levantado: {both_arms_raised_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Braco Direito levantado: {right_arm_raised_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Braco Esquerdo levantado: {left_arm_raised_count}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Movimentos de Maos: {wave_count + handshake_count}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Movimentos de Cabeca: {nod_count}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Mao no Rosto: {hand_near_face_count}', (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)

        # write frame to output video
        out.write(frame)

    anomalos_count = len(set(activities))

    with open('summaries/activity_recognition_summary.txt', 'w') as f:
        f.write(f"Identificao de Gestos e Movimentos - Total de frames analisados: {total_frames}\n")
        f.write(f"Identificao de Gestos e Movimentos - Ambos os Bracos levantados: {both_arms_raised_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Apenas o Braco Direito levantado: {right_arm_raised_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Apenas o Braco Esquerdo levantado: {left_arm_raised_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos de Maos: {wave_count + handshake_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos de Cabeca: {nod_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Mao no Rosto: {hand_near_face_count}\n")
        f.write(f"Identificao de Gestos e Movimentos - Movimentos Anomalos: {anomalos_count}\n")

    # release resources
    cap.release()
    out.release()
    print("Processamento concluído.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_asset_path = os.path.join(script_dir, 'models/gesture_recognizer.task')
    VisionRunningMode = vision.RunningMode.VIDEO
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode,
        num_hands=2,
        min_tracking_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_hand_detection_confidence=0.3
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    recognize_activities(recognizer)