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

    #img = greenmask(img)                                                               # Se extrae una mascara de zonas verdes
    #cv2.imshow('Imagen filtro 4',img)

    bordes = cv2.Canny(img,50,200,3, L2gradient=True)                                   # Se detectan los bordes de la imagen
    cv2.imshow('Deteccion de Bordes',bordes)                                            # Se muestran los bordes detectados
    
    DeteccionParcelas(bordes, original)                                                 # Se aplica la transformada de Hough para la deteccion de los bordes de las parcelas
    cv2.waitKey(0)
   
    cv2.destroyAllWindows()

    
def kmeans(img):

    global k                                                                            #Variable global que representa la cantidad de centros del kmeans
    
    Z = img.reshape((-1,3))                                                             #El frame a analizar se vuelve una matriz con la cantidad de pixeles y sus canales de color
    Z = np.float32(Z)                                                                   #Se asegura que los valroes de la matriz sean tipo float para su manipulacion mediante la 
                                                                                        #libreria numpy
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)           #Se definen los criterios a utilizar con openCV para el uso de cv2.kmeans()
                                                                                        #Donde cv2.TERM_CRITERIA_EPS se refiere al erro valido por los datos y 
                                                                                        #cv2.TERM_CRITERIA_MAX_ITER la cantidad de iteraciones maxima
    
    ret,label,center = cv2.kmeans(Z,k,None,criteria,10,cv2.KMEANS_PP_CENTERS)           #funcion de openCV que genera clusters tipo kmeans de los datos indicados
                                                                                        #Los datos se presentan en Z, k es la cantidad de centros, 10 se refiere a la cantidad de
                                                                                        #Intentos que la funcion va a ejecutarse para buscar el mejor resultado y
                                                                                        #cv2.KMEANS_PP_CENTERS se refiere al metodo de seleccion de los centros, en este caso busca
                                                                                        #por un metodo probabilistico los mejores centros pero aumenta el costo computacional
    
    center = np.uint8(center)                                                           #Se transforman los centros a enteros para porceder a obtener la nueva imagen
    res = center[label.flatten()]                                                       #Se vuelve a crear la imagen 2d y se le da las mismas caracteristicas que el frame original
    res2 = res.reshape((img.shape))
    
    return res2


def DeteccionParcelas(bordes, original):                                                # Funcion principal de deteccion de parcelas por medio de la transformada de Hough
       
    lines = cv2.HoughLinesP(bordes, 1, np.pi/180, 50, 10, 50, 5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(original, (x1, y1), (x2, y2), (20, 220, 20), 3)                    # Se dibujan los limites detectados en cada parcela

    cv2.imshow('Parcelas Detectadas',original)                                          # Se muestra el resultado final

def greenmask(img):                                                                     # Funcion de obtencion de la mascara de verdes

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                                          # Se transforma la imagen a espacio HSV
    low_green = np.array([25, 52, 72])                                                  # Se definen los limites del color verde
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, low_green, high_green)                                # Se crea una mascara con los pixeles dentro del rango
    img = cv2.bitwise_and(img, img, mask=green_mask)                                    # Se aplica un AND con la imagen original y la mascara generada

    return img

main()
    

    

