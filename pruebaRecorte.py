import cv2

# Ruta de la imagen a probar
ruta_img = r"C:\MisArchivos\Escritorio\ClasificadorFrames\dataset\ejemplo.png"
img = cv2.imread(ruta_img)

if img is None:
    print("❌ No se pudo cargar la imagen.")
    exit()

# Mostrar la imagen y permitir seleccionar ROI
roi = cv2.selectROI("Selecciona el área del hocico", img, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = roi
print(f"✅ Coordenadas seleccionadas: x={x}, y={y}, ancho={w}, alto={h}")

# Guardar imagen recortada para ver el resultado
recorte = img[y:y+h, x:x+w]
cv2.imwrite("recorte_preview.png", recorte)
print("💾 Imagen recortada guardada como 'recorte_preview.png'")
