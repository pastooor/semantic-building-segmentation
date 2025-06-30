import os
import numpy as np
from PIL import Image

def cortar_imagen(imagen, tamaño_bloque=1024, solapamiento=30):
    """
    Divide la imagen en bloques de 1024x1024 con un solapamiento de 30 píxeles en todas las direcciones.
    Se asegura de cubrir toda la imagen, incluyendo los bordes.
    """
    img_width, img_height = imagen.size
    bloques = []
    
    desplazamiento = tamaño_bloque - solapamiento
    
    for i in range(0, img_width, desplazamiento):
        for j in range(0, img_height, desplazamiento):
            if i + tamaño_bloque > img_width:
                i = img_width - tamaño_bloque
            if j + tamaño_bloque > img_height:
                j = img_height - tamaño_bloque

            caja = (i, j, i + tamaño_bloque, j + tamaño_bloque)
            bloque = imagen.crop(caja)
            bloques.append((bloque, i, j))

    return bloques

def guardar_bloques(bloques, carpeta_destino, nombre_imagen):
    """
    Guarda los bloques en la carpeta destino con nombre basado en la posición.
    """
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    for idx, (bloque, i, j) in enumerate(bloques):
        nombre_bloque = f"{nombre_imagen}_pos_{i}_{j}.tif"
        bloque.save(os.path.join(carpeta_destino, nombre_bloque))

        print(f"Guardado: {nombre_bloque}")

def procesar_train(carpeta_origen, carpeta_origen_mask, carpeta_destino_img, carpeta_destino_mask):
    """
    Procesa las imágenes de entrenamiento junto con sus máscaras.
    """
    for file in os.listdir(carpeta_origen):
        if file.endswith('.tif'):
            ruta_imagen = os.path.join(carpeta_origen, file)
            ruta_mascara = os.path.join(carpeta_origen_mask, file)

            if not os.path.exists(ruta_mascara):
                continue  

            imagen = Image.open(ruta_imagen)
            mascara = Image.open(ruta_mascara).convert("L")  

            bloques = cortar_imagen(imagen)
            bloques_mascara = cortar_imagen(mascara)

            nombre_imagen = file.split('.')[0]
            bloques_validos = [
                (b, bm, i, j) for (b, i, j), (bm, _, _) in zip(bloques, bloques_mascara)
            ]

            for bloque, bloque_mascara, i, j in bloques_validos:
                if not os.path.exists(carpeta_destino_img):
                    os.makedirs(carpeta_destino_img)
                if not os.path.exists(carpeta_destino_mask):
                    os.makedirs(carpeta_destino_mask)

                nombre_bloque = f"{nombre_imagen}_pos_{i}_{j}.tif"
                bloque.save(os.path.join(carpeta_destino_img, nombre_bloque))
                bloque_mascara.save(os.path.join(carpeta_destino_mask, nombre_bloque))

                print(f"Guardado: {nombre_bloque}")

def procesar_test(carpeta_origen, carpeta_destino):
    """
    Procesa las imágenes de test (sin máscaras).
    """
    for file in os.listdir(carpeta_origen):
        if file.endswith('.tif'):
            ruta_imagen = os.path.join(carpeta_origen, file)
            imagen = Image.open(ruta_imagen)

            bloques = cortar_imagen(imagen)

            nombre_imagen = file.split('.')[0]
            guardar_bloques(bloques, carpeta_destino, nombre_imagen)

def main():
    carpeta_imagenes_train = "AerialImageDataset/train/images"
    carpeta_masks_train = "AerialImageDataset/train/gt"
    carpeta_destino_train_images = "processed_images/train/images"
    carpeta_destino_train_masks = "processed_images/train/gt"

    carpeta_imagenes_test = "AerialImageDataset/test/images"
    carpeta_destino_test_images = "processed_images/test/images"

    procesar_train(carpeta_imagenes_train, carpeta_masks_train, carpeta_destino_train_images, carpeta_destino_train_masks)

    procesar_test(carpeta_imagenes_test, carpeta_destino_test_images)

if __name__ == "__main__":
    main()

