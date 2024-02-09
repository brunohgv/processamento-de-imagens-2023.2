import cv2
import imutils
import numpy as np
import easyocr
import re

THRESH_MIN = 90
THRESH_MAX = 255
plate_regex = re.compile("^[a-zA-Z]{3}[0-9][A-Za-z0-9][0-9]{2}$")

# Carrega imagem
image = cv2.imread("./images/carro-4.jpeg")


# Transforma para tons de cinza para conseguir binzarizar (cores não importam para a detecção)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Binariza a imagem para conseguir identificar melhor as diferenças de tom
_, thresholded = cv2.threshold(image_gray, THRESH_MIN, THRESH_MAX, cv2.THRESH_BINARY)
# Aplica um filtro bilateral para reduzir o ruído
bfilter = cv2.bilateralFilter(image_gray, 11, 17, 17)
# Identifica arestas da imagem por contraste
edged = cv2.Canny(bfilter, 30, 200)
# Encontra os contornos da imagem aproximando os valores
keypoints = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
# Ordena pela área do contorno da maior para a menor
contours = sorted(contours, key=cv2.contourArea, reverse=True)
# Filtra pelas 10 maiores áreas
contours = contours[:10]

location = None

# Encontra um polígono similar a um retângulo
for contour in contours:
  approx = cv2.approxPolyDP(contour, 10, True)
  if len(approx) == 4: # se o polígono tem 4 lados
    location = approx
    break

# Cria uma máscara de pixels 0 (pretos)
mask = np.zeros(image_gray.shape, np.uint8)
# Cria uma nova imagem com pixels brancos usando o contorno como borda
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
# Faz uma operação and para deixar apenas a placa (area do contorno) aparente
new_image = cv2.bitwise_and(image, image, mask=mask)

# Pega todos os pixels brancos da máscara
(x,y) = np.where(mask==255)
# Pega a coordenada mínima
(x1, y1) = (np.min(x), np.min(y))
# Pega a coordenada máxima
(x2, y2) = (np.max(x), np.max(y))
# Corta a imagem nessas coordenadas
cropped = image_gray[x1:x2, y1:y2]

# Usa o easyocr para ler a imagem resultante
reader = easyocr.Reader(['pt'])
results = reader.readtext(cropped)
# Filtra apenas formatos válidos de placas para evitar pegar texto residual
valid_plates = [ result[1] for result in results if plate_regex.match(result[1]) ]

text = valid_plates[0]
font = cv2.FONT_HERSHEY_SIMPLEX
result_image = cv2.putText(image, text=text, org=(approx[0][0][0], approx[0][0][1]), fontFace=font, fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)
result_image = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0,0,255), 3)

cv2.imshow("Imagem Resultante", result_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

