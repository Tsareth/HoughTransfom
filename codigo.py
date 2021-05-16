import numpy as np                          # Importar librerias necesarias
import cv2
import keyboard
import imutils
import time
   

global k
k = 4
global name
global width
global height

def main():
    global name
    global width
    global height
    name = input('Ingrese el nombre de la imagen a analizar')

    img = cv2.imread(name,cv2.IMREAD_COLOR)

    width = img.shape[1]
    height = img.shape[0]
    
    img = cv2.resize(img,(int(width/2),int(height/2)),interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Imagen original',img)
    img = cv2.bilateralFilter(img,15,75,75)
    cv2.imshow('Imagen Filtrada',img)
    
    img_kmeans = kmeans(img)
    cv2.imshow('Imagen Kmeans',img_kmeans)

    DeteccionParcelas(img, img_kmeans)
    cv2.waitKey(0)

    
    
    cv2.destroyAllWindows()

    
def kmeans(img):

    global k                                                                            #Variable global que representa la cantidad de centros del kmeans
    
    Z = img.reshape((-1,3))                                                           #El frame a analizar se vuelve una matriz con la cantidad de pixeles y sus canales de color
    Z = np.float32(Z)                                                                   #Se asegura que los valroes de la matriz sean tipo float para su manipulacion mediante la 
                                                                                        #libreria numpy
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)            #Se definen los criterios a utilizar con openCV para el uso de cv2.kmeans()
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
    
    #frame_resized = cv2.resize(res2,
                               #(int(width_original),int(height_original)),
                               #fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    return res2


def DeteccionParcelas(img,img_kmeans):
    
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##gray = np.float32(grayscale) #Convert numeric type
 
    ##kernel = np.ones((5,5),np.float32)/25
    #dst = cv2.filter2D(gray,-1,kernel)
    #dst = np.uint8(dst)
    ##dst = dst.reshape((grayscale.shape))
    
    #cv2.imshow('Gray',dst)
    #cv2.imshow('Grayscale',grayscale)
    bordes = cv2.Canny(grayscale,140,200)
    cv2.imshow('Bordes',bordes)


##    lines = cv2.HoughLines(bordes,1,np.pi/180,170)
##    for i in range(0, len(lines)):
##        
##        rho = lines[i][0][0]
##        theta = lines[i][0][1]
##        a = np.cos(theta)
##        b = np.sin(theta)
##        x0 = a*rho
##        y0 = b*rho
##        x1 = int(x0 + 1000*(-b))
##        y1 = int(y0 + 1000*(a))
##        x2 = int(x0 - 1000*(-b))
##        y2 = int(y0 - 1000*(a))
##
##        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    
    
    lines = cv2.HoughLinesP(bordes, 1, np.pi/180, 60, np.array([]), 50, 5)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (20, 220, 20), 3)

    cv2.imshow('Parcelas Detectadas',img)
    
    
    
    
main()
    

    

