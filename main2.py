import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

archivo = 'src/tirada_1.mp4'
archivo_salida = 'resultado_tirada.mp4'

os.makedirs("frames", exist_ok = True)

#lectura
cap = cv2.VideoCapture(archivo)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(width)
print(height)
print(fps)
print(n_frames)

frame_prev = None
cont_fram_est = 0
cont_frame_global = 0
umbral_mov = 2.0
frames_estaticos = 0

#analiza si hay imagen quieta y guarda el frame estatico
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    #detectar movimiento
    es_estatico = False
    if frame_prev is not None:
        frame_diff = cv2.absdiff(frame_prev, blur)   #calcular diferencia entre frames
        motion_score = np.mean(frame_diff)

        if motion_score < umbral_mov:
            frames_estaticos += 1
        else:
            frames_estaticos = 0

        if frames_estaticos > 25:    # quieto 25 frames seguidos
            es_estatico = True
            nombre_archivo = f"frames/frame_{cont_frame_global}.jpg" #guardar frame quieto
            cv2.imwrite(nombre_archivo, frame)
            cont_frame_global += 1

    frame_prev = blur

print(cont_frame_global)

cap.release()
cv2.destroyAllWindows()

archivos_videos = [
    'src/tirada_1.mp4',
    'src/tirada_2.mp4',
    'src/tirada_3.mp4',
    'src/tirada_4.mp4'
]

AREA_MINIMA = 500
AREA_MAXIMA = 30000
rojo_bajo_1 = np.array([0, 80, 100])
rojo_alto_1 = np.array([10, 255, 255])
rojo_bajo_2 = np.array([170, 80, 100])
rojo_alto_2 = np.array([180, 255, 255])
margen = 15


for archivo in archivos_videos:
    print(f"\n Video: {archivo} ")
    nombre_base = os.path.splitext(os.path.basename(archivo))[0]
    carpeta_frames = os.path.join("frames", nombre_base)
    os.makedirs(carpeta_frames, exist_ok=True)

    cap = cv2.VideoCapture(archivo)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    frame_prev = None
    cont_frame_global = 0
    umbral_mov = 2.0
    frames_estaticos = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Detectar movimiento
        if frame_prev is not None:
            frame_diff = cv2.absdiff(frame_prev, blur)
            motion_score = np.mean(frame_diff)

            if motion_score < umbral_mov:
                frames_estaticos += 1
            else:
                frames_estaticos = 0

            if frames_estaticos > 18:    #quieto 25 frames seguidos+
                nombre_frame = os.path.join(carpeta_frames, f"frame_{cont_frame_global}.jpg")
                cv2.imwrite(nombre_frame, frame)
                cont_frame_global += 1

                #break #lo agregue para que solo guarde un frame

        frame_prev = blur

    cap.release()
    print(f"Frames estáticos: {cont_frame_global}")

    #parte2

    lista_archivos_img = os.listdir(carpeta_frames)

    lista_archivos_img = [f for f in lista_archivos_img if f.endswith('.jpg')]


    for nombre_archivo_img in lista_archivos_img:
        ruta_completa = os.path.join(carpeta_frames, nombre_archivo_img)
        imagen = cv2.imread(ruta_completa)

        if imagen is None:
            continue

        imagen_dibujo = imagen.copy()

        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, rojo_bajo_1, rojo_alto_1)
        mask2 = cv2.inRange(hsv, rojo_bajo_2, rojo_alto_2)
        imagen_binaria = mask1 + mask2

        kernel = np.ones((9, 9), np.uint8)
        imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)
        imagen_binaria = cv2.dilate(imagen_binaria, kernel, iterations=1)
        imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        dados_encontrados = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if AREA_MINIMA < area < AREA_MAXIMA:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h

                if 0.5 < aspect_ratio < 1.5:
                    cv2.rectangle(imagen_dibujo, (x, y), (x+w+5, y+h+5), color=(255, 0, 0), thickness=2)
                    dados_encontrados += 1

                    y_fin = min(imagen.shape[0], y+h+margen)
                    x_fin = min(imagen.shape[1], x+w+margen)

                    roi = imagen_dibujo[y : y_fin, x : x_fin]

                    if roi.size == 0: continue

                    roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur_roi = cv2.GaussianBlur(roi_gris, (3, 3), 0)
                    #_, thresh = cv2.threshold(blur_roi, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    thresh = cv2.adaptiveThreshold(blur_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 15, 3)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
                    cnts_puntos, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    puntaje = 0
                    for c in cnts_puntos:
                        area_p = cv2.contourArea(c)
                        perim_p = cv2.arcLength(c, True)
                        if perim_p == 0: continue

                        # Circularidad y tamaño para detectar los puntos blancos
                        if (50 < area_p < roi.size * 0.05) and ((4 * np.pi * area_p) / (perim_p ** 2) > 0.75):
                            puntaje += 1

                    texto = f'D{dados_encontrados}: {str(puntaje)}p'
                    cv2.putText(imagen_dibujo, texto, (x - 200, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

        if dados_encontrados >= 1:
            plt.figure()
            plt.title(f"Video: {nombre_base} | Frame: {nombre_archivo_img}")
            plt.imshow(cv2.cvtColor(imagen_dibujo, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show(block=False)

    cv2.destroyAllWindows()

archivos_videos = [
    'src/tirada_1.mp4',
    'src/tirada_2.mp4',
    'src/tirada_3.mp4',
    'src/tirada_4.mp4'
]

AREA_MINIMA = 500
AREA_MAXIMA = 30000

rojo_bajo_1 = np.array([0, 95, 100])
rojo_alto_1 = np.array([10, 255, 255])
rojo_bajo_2 = np.array([170, 95, 100])
rojo_alto_2 = np.array([180, 255, 255])

margen = 15

os.makedirs("resultados_videos", exist_ok=True)

for archivo in archivos_videos:
    cap = cv2.VideoCapture(archivo)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    nombre_salida = os.path.join("resultados_videos", f"resultado_{os.path.basename(archivo)}")
    out = cv2.VideoWriter(nombre_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_prev = None
    frames_estaticos = 0
    umbral_mov = 2.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_salida = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        if frame_prev is not None:
            frame_diff = cv2.absdiff(frame_prev, blur)
            motion_score = np.mean(frame_diff)

            if motion_score < umbral_mov:
                frames_estaticos += 1
            else:
                frames_estaticos = 0

            if frames_estaticos > 13:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(hsv, rojo_bajo_1, rojo_alto_1)
                mask2 = cv2.inRange(hsv, rojo_bajo_2, rojo_alto_2)
                imagen_binaria = mask1 + mask2

                kernel_dados = np.ones((9, 9), np.uint8)
                imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel_dados)
                imagen_binaria = cv2.dilate(imagen_binaria, kernel_dados, iterations=1)
                imagen_binaria = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel_dados)

                contours, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                dados_encontrados = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)

                    if AREA_MINIMA < area < AREA_MAXIMA:
                        x, y, w, h = cv2.boundingRect(cnt)
                        aspect_ratio = w / h

                        if 0.5 < aspect_ratio < 1.5:
                            cv2.rectangle(frame_salida, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            dados_encontrados += 1

                            y_fin = min(frame.shape[0], y+h+margen)
                            x_fin = min(frame.shape[1], x+w+margen)
                            roi = frame[y : y_fin, x : x_fin]

                            if roi.size == 0: continue

                            roi_gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                            blur_roi = cv2.GaussianBlur(roi_gris, (3, 3), 0)

                            thresh = cv2.adaptiveThreshold(blur_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv2.THRESH_BINARY_INV, 15, 3)

                            kernel_puntos = np.ones((2,2), np.uint8)
                            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_puntos)

                            cnts_puntos, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                            puntaje = 0
                            for c in cnts_puntos:
                                area_p = cv2.contourArea(c)
                                perim_p = cv2.arcLength(c, True)
                                if perim_p == 0: continue

                                if (50 < area_p < roi.size * 0.05) and ((4 * np.pi * area_p) / (perim_p ** 2) > 0.70): # Bajé un poco la circularidad a 0.70 por si el ángulo deforma el círculo
                                    puntaje += 1

                            texto = f'D{dados_encontrados}: {puntaje}'
                            cv2.putText(frame_salida, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        frame_prev = blur
        out.write(frame_salida)

    cap.release()
    out.release()

cv2.destroyAllWindows()