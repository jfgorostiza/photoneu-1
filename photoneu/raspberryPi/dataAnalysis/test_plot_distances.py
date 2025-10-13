import cv2
import numpy as np
import matplotlib.pyplot as plt

# Crear imagen binaria con varios blobs solapados
img = np.zeros((300, 400), dtype=np.uint8)
cv2.circle(img, (100, 150), 60, 255, -1)
cv2.circle(img, (160, 150), 60, 255, -1)
cv2.circle(img, (300, 180), 70, 255, -1)

# Encontrar contornos
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

plt.figure(figsize=(10,6))

for idx, cnt in enumerate(contours):
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    distancias = np.zeros(len(cnt))  # vector con la profundidad para cada punto del contorno

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, depth = defects[i,0]
            profundidad = depth / 256.0
            # asignamos la profundidad al índice "far"
            distancias[f] = profundidad
    
    # Representar curva
    plt.plot(range(len(cnt)), distancias, label=f'Blob {idx+1}')

plt.title("Curvas de profundidad de convexityDefects por contorno")
plt.xlabel("Índice del punto en el contorno")
plt.ylabel("Profundidad (px)")
plt.legend()
plt.grid(True)
plt.show()
