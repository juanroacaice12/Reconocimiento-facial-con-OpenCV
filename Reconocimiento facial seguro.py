import cv2
import logging

# Configuración de registro de actividad
logging.basicConfig(filename='activity_log.txt', level=logging.INFO)

# Configuración de la ruta del clasificador
face_cascade_path = 'python devolper/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Configuración de la cámara
cap = cv2.VideoCapture(0)

try:
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Registro de actividad
        logging.info(f'Detección de rostros: {len(faces)}')

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('img', img)
        k = cv2.waitKey(30)
        if k == 27:
            break

except Exception as e:
    print(f"Error: {str(e)}")

finally:
    # Liberación de la captura de video y cierre de ventanas
    cap.release()
    cv2.destroyAllWindows()
