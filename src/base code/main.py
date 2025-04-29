import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab = a - b
    cb = c - b
    rad = np.arccos(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)))
    return np.degrees(rad)

cv2.namedWindow("Detector de Queda", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detector de Queda", 1280, 720)

altura_cabeca_anterior = None
tempo_anterior = None
velocidade_queda = 0
queda_detectada = False  # <----- NOVO

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

        # Coletar coordenadas dos pontos para caixa
        pontos_x = [int(p.x * w) for p in lm]
        pontos_y = [int(p.y * h) for p in lm]
        min_x, max_x = min(pontos_x), max(pontos_x)
        min_y, max_y = min(pontos_y), max(pontos_y)

        # Desenha a caixa preta ao redor do corpo
        cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 0, 0), 3)

        # Posições para cálculo do ângulo e inclinação do tronco
        ombro = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                 lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        quadril = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        joelho = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        cabeca_y = lm[mp_pose.PoseLandmark.NOSE.value].y

        angulo = calcular_angulo(ombro, quadril, joelho)

        # Cálculo da inclinação do tronco (ombro -> quadril)
        tronco_dx = ombro[0] - quadril[0]
        tronco_dy = ombro[1] - quadril[1]
        angulo_tronco = np.degrees(np.arctan2(tronco_dy, tronco_dx))
        angulo_tronco = abs(angulo_tronco)

        # Cálculo da velocidade de queda
        tempo_atual = time.time()
        if altura_cabeca_anterior is not None and tempo_anterior is not None:
            dt = tempo_atual - tempo_anterior
            velocidade_queda = (cabeca_y - altura_cabeca_anterior) / dt

        altura_cabeca_anterior = cabeca_y
        tempo_anterior = tempo_atual

        # Classificação da postura
        if velocidade_queda > 0.2 and angulo < 100:
            estado = "Queda Detectada!"
            cor = (0, 0, 255)  # Vermelho
            queda_detectada = True  # <----- QUEDA FOI DETECTADA
        elif angulo > 140:
            estado = "Em pe"
            cor = (0, 0, 0)  # Preto
        elif 90 < angulo <= 140 and 70 <= angulo_tronco <= 110:
            estado = "Sentado"
            cor = (0, 255, 0)  # Verde
        else:
            estado = "Deitado"
            cor = (255, 0, 255)  # Roxo

        # Escreve o estado acima da caixa
        cv2.putText(frame, f"{estado}", (min_x, min_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 3)

    # Se uma queda já foi detectada, mostrar a mensagem sempre
    if queda_detectada:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)  # Tela Vermelha
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        texto = "QUEDA DETECTADA!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 10
        tamanho_texto, _ = cv2.getTextSize(texto, font, font_scale, thickness)
        text_x = (frame.shape[1] - tamanho_texto[0]) // 2
        text_y = (frame.shape[0] + tamanho_texto[1]) // 2
        cv2.putText(frame, texto, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Ajuste para exibição em tela
    win_x, win_y, win_w, win_h = cv2.getWindowImageRect("Detector de Queda")
    frame_resized = cv2.resize(frame, (win_w, win_h))

    cv2.imshow("Detector de Queda", frame_resized)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
