import cv2
import numpy as np

# Crear imagen negra
img = np.zeros((500, 500, 3), dtype=np.uint8)

# Generar puntos de un arco (ejemplo: 120 grados de un círculo)
center = (250, 250)
radius = 100
angle_start = 0
angle_end = 120
points = []

for a in range(angle_start, angle_end, 2):
    x = int(center[0] + radius * np.cos(np.deg2rad(a)))
    y = int(center[1] + radius * np.sin(np.deg2rad(a)))
    points.append([x, y])

contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

# Dibujar contorno
cv2.polylines(img, [contour], isClosed=False, color=(255, 255, 255), thickness=2)

# Ajustar rectángulo mínimo
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect).astype(int)
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
rect_center = tuple(map(int, rect[0]))
cv2.circle(img, rect_center, 5, (0, 255, 0), -1)

# Ajustar elipse (requiere >=5 puntos)
if len(contour) >= 5:
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(img, ellipse, (255, 0, 0), 2)
    ellipse_center = tuple(map(int, ellipse[0]))
    cv2.circle(img, ellipse_center, 5, (255, 0, 0), -1)

# Mostrar resultados
cv2.imshow("Comparacion rectangulo vs elipse", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
