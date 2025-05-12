import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime, timedelta

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    rad = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    return np.degrees(rad)

def enviar_notificacao_celular():
    print("Notificação enviada: Queda detectada!")

cv2.namedWindow("Detector de Queda", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detector de Queda", 1280, 720)

altura_quadril_anterior = None
tempo_anterior = None
velocidade_queda = 0
queda_detectada = False
tempo_queda = None
notificacao_enviada = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)
    estado = "Desconhecido"

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        h, w, _ = frame.shape

        pontos_x = [int(p.x * w) for p in lm]
        pontos_y = [int(p.y * h) for p in lm]
        min_x, max_x = min(pontos_x), max(pontos_x)
        min_y, max_y = min(pontos_y), max(pontos_y)

        cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 0, 0), 3)

        ombro = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        quadril = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        joelho = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        quadril_y = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h

        angulo = calcular_angulo(ombro, quadril, joelho)
        tronco_dx = ombro[0] - quadril[0]
        tronco_dy = ombro[1] - quadril[1]
        angulo_tronco = abs(np.degrees(np.arctan2(tronco_dy, tronco_dx)))

        tempo_atual = time.time()
        if altura_quadril_anterior is not None and tempo_anterior is not None:
            dt = tempo_atual - tempo_anterior
            velocidade_queda = (quadril_y - altura_quadril_anterior * h) / dt

        altura_quadril_anterior = lm[mp_pose.PoseLandmark.LEFT_HIP.value].y
        tempo_anterior = tempo_atual

        corpo_perto_chao = quadril_y > h * 0.75
        tronco_inclinado = angulo < 100 and angulo_tronco > 60
        queda_rapida = velocidade_queda > 100

        if corpo_perto_chao and tronco_inclinado and queda_rapida:
            estado = "Queda Detectada!"
            cor = (0, 0, 255)
            if not queda_detectada:
                queda_detectada = True
                tempo_queda = datetime.now()
                if not notificacao_enviada:
                    enviar_notificacao_celular()
                    notificacao_enviada = True
        elif angulo > 140:
            estado = "Em pé"
            cor = (0, 0, 0)
        elif 90 < angulo <= 140 and 70 <= angulo_tronco <= 110:
            estado = "Sentado"
            cor = (0, 255, 0)
        else:
            estado = "Deitado"
            cor = (255, 0, 255)

        cv2.putText(frame, f"{estado}", (min_x, min_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 3)

        if queda_detectada and tempo_queda:
            if datetime.now() - tempo_queda <= timedelta(minutes=30):
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
                cv2.putText(frame, "QUEDA DETECTADA!", (50, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                queda_detectada = False
                notificacao_enviada = False

    win_x, win_y, win_w, win_h = cv2.getWindowImageRect("Detector de Queda")
    frame_resized = cv2.resize(frame, (win_w, win_h))

    cv2.imshow("Detector de Queda", frame_resized)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Finalizando o programa
# O código foi finalizado com sucesso.
