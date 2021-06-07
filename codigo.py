import numpy as np                                                                      # Se importan las librerias necesarias
import cv2
import keyboard
import imutils
import time
   
global k                                                                                # Se definen las variables globaes necesarias
k = 8                                                                                   # Numero de centroides para K-means
global name
global width
global height

def main():                                                                             # Definicion de la rutina principal
    global name
    global width
    global height
    name = input('Ingrese el nombre de la imagen a analizar')                           # Se pide al usuario un nombre de archivo
    #name = "Imagen.jpeg"
    
    img = cv2.imread(name,cv2.IMREAD_COLOR)                                             # Se abre el archivo seleccionado

    width = img.shape[1]                                                                # Se obtienen las dimensiones de la imagen
    height = img.shape[0]
    
    img = cv2.resize(img,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)  # Se reduce el tamaño de la imagen a la mitad
    original = img                                                                      # Se guarda una copia del original de la imagen
    cv2.imshow('Imagen Original',img)                                                   # Se muestra el original de la imagen
    
    img = cv2.bilateralFilter(img,9,150,150)                                            # Se aplica un filtro bilateral para deducir ruido y conservar bordes
    #cv2.imshow('Imagen filtro 3',img)

    bordes = cv2.Canny(img,50,200,3, L2gradient=True)                                   # Se detectan los bordes de la imagen
    cv2.imshow('Deteccion de Bordes',bordes)                                            # Se muestran los bordes detectados
    
    DeteccionParcelas(bordes, original)                                                 # Se aplica la transformada de Hough para la deteccion de los bordes de las parcelas
    cv2.waitKey(0)
   
    cv2.destroyAllWindows()

def DeteccionParcelas(bordes, original):                                                # Funcion principal de deteccion de parcelas por medio de la transformada de Hough
       
    lines = cv2.HoughLinesP(bordes, 1, np.pi/180,50,minLineLength=50,maxLineGap=5)      # Transformada de Hough
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(original, (x1, y1), (x2, y2), (20, 220, 20), 3)                    # Se dibujan los limites detectados en cada parcela

    cv2.imshow('Parcelas Detectadas',original)                                          # Se muestra el resultado final

main()
    

    

