# Datasheet – NumPy y Pandas

## Sección A – NumPy: Creación y manipulación de arreglos
    -Rama: A-numpy-arreglos  

```python
#Ejemplos realizados y ejecutados en google colab( antes de todo, cabe aclarar que en el google colab evidentemente primero se importo el Numpy y se revisó la versión la cual es la 2.0.2):

    # 1. Crear un arreglo con "array"
        #Con esto se crea un arreglo con arrays:
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print("Arreglo:", arr)

    #Salida:
        Arreglo: [1 2 3 4 5]

    # 2. Crear un arreglo con "arange y linspace"
        #Con esto creamos un arreglo con range y linspace (esto sirve para hacer secuancias y que despues podamos pues hacer el arreglo de forma automatica, donde se coloica primero el inicio, luego el fin y luego en cuanto va a ir la secuencia)(arange = enteros, linspace = flotantes): 
        a = np.arange(0, 10, 2)
        b = np.linspace(0, 1, 5)

        print("Arange:", a)
        print("Tipo:", a.dtype)

        print("Linspace:", b)
        print("Tipo:", b.dtype)

    #Salida:
        Arange: [0 2 4 6 8]
        Tipo: int64
        Linspace: [0.   0.25 0.5  0.75 1.  ]
        Tipo: float64

    # 3. Cambiar forma con "reshape"
        #Con reshape podemos cambiarle al forma al arreglo, de esta forma:
        m = np.arange(6).reshape(2, 3)
        print("Reshape 2x3:\n", m)
        print("Dimensiones:", m.ndim)

        m2 = np.arange(4).reshape(2, 2)
        print("Reshape 2x2:\n", m2)
        print("Dimensiones:", m2.ndim)

    #Salida:
        Reshape 2x3:
        [[0 1 2]
        [3 4 5]]
        Dimensiones: 2

        Reshape 2x2:
        [[0 1]
        [2 3]]
        Dimensiones: 2

    # 4. Concatenar arreglos
        #Ahora con esto podremos unir o concatar los arrays:
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        cat = np.concatenate((x, y))

        print("Arreglo x:", x)
        print("Arreglo y:", y)
        print("Concatenado:", cat)

    #Salida:
        Arreglo x: [1 2 3]
        Arreglo y: [4 5 6]
        Concatenado: [1 2 3 4 5 6]

    # 5. Operaciones básicas
        #a) Con un solo arreglo
            #Esta es una forma de usar operaciones basicas con los arreglos:
            z = np.array([1, 2, 3, 4])
            print("Array:", z)
            print("Suma +10:", z + 10)
            print("Multiplicación x2:", z * 2)
            print("Potencia al cuadrado:", z ** 2)

        #Salida:
            Array: [1 2 3 4]
            Suma +10: [11 12 13 14]
            Multiplicación x2: [2 4 6 8]
            Potencia al cuadrado: [ 1  4  9 16]

        #b) Operaciones entre dos arreglos
            #Esta es otra forma, donde operamos dos arrays:
            Array1 = np.array([1, 2, 3, 4])
            Array2 = np.array([5, 6, 7, 8])

            print("Array1:", Array1)
            print("Array2:", Array2)
            print("Suma de los arrays:", Array1 + Array2)
            print("Multiplicación entre los arrays:", Array1 * Array2)
            print("Array1 elevado al Array2:", Array1 ** Array2)

        #Salida:
            Array1: [1 2 3 4]
            Array2: [5 6 7 8]
            Suma de los arrays: [ 6  8 10 12]
            Multiplicación entre los arrays: [ 5 12 21 32]
            Array1 elevado al Array2: [     1     64  2187 65536]

```
## Sección B – NumPy Estadísticas y Avanzado 
    -Rama: B-numpy-estadisticas

##### 1) Comando mean (media)
Calcula el promedio de los elementos en un array 

Entrada:
    
    pruebamean = np.array([10, 20, 30, 40, 50, 60, 70])
     print("La media del Array pruebamean es:",np.mean(pruebamean))

#salida: 

    La media del Array pruebamean es: 40.0

##### 2) Comando stg (desviación estándar)
Mide la desviación de los datos respecto al promedio.

Entrada:
    
    pruebastg = np.array([10, 20, 30, 40, 50, 60, 70])
    print("La desviación estándar del Array pruebastd es:",np.std(pruebastd))
#salida
    
    La desviación estándar del Array pruebastg es: 20.0

##### 3) Comando sum (Suma elementos)
Realiza la suma de todos los elementos

Entrada:
    
    pruebasum = np.array([10, 20, 30, 40, 50, 60, 70])
    print("La suma de elementos del Array pruebasum es:",np.sum(pruebasum))
Salida:

    La suma de elementos del Array pruebasum es: 280

##### 4) Comando arange 
Genera un array con una secuencia de números espaciados regularmente por un valor definido.

###### Estructura: np.arange(inicio, final, paso)
- inicio: Valor en el que inicia la secuencia
- final: Valor límite  de la secuencia (no llega a este número)
- Paso: tamaño del intervalo entre valores

Entrada:
    
    print("La secuencia generada cada 2 números de 0 hasta el límite de 20 es: \n",np.arange(0, 20, 2))

Salida:
    
    La secuencia generada cada 2 números de 0 hasta el límite de 20 es: 
     [ 0  2  4  6  8 10 12 14 16 18]



##### 5) Comando linspace
Genera un arreglo de números equidistantes entre dos valores.

###### Estructura: np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
- start: valor inicial del intervalo.
- stop: valor final del intervalo.
- num: número de puntos que quieres generar (por defecto 50).
- endpoint: si es True, incluye el valor final (stop) en el arreglo.
- rtstep: si es True, además del arreglo, devuelve el paso entre los valores.
- dtype: tipo de dato (float, int, etc.).

Entrada:
    
    arr = np.linspace(0, 10, num=5)
    print("El arreglo generado con 5 puntos de 0 a 10 es: ",arr)

Salida:
    
    El arreglo generado con 5 puntos de 0 a 10 es:  [ 0.   2.5  5.   7.5 10. ]
    
##### 6) Comandos random
Np.random es el módulo clásico de NumPy para generar números aleatorios. Permite crear datos simulados, mezclar arreglos, y trabajar con distribuciones estadísticas.

###### - np.random.rand() 
 Genera un número aleatorio entre 0 y 1

 Estructura: random.rand(num)
- num: cantidad de números entre 0 y 1 a generar.

Entrada:

    print("Este comando genera 4 números aleatorios entre 0 y 1: ",np.random.rand(4) )
Salida:

    Este comando genera 4 números aleatorios entre 0 y 1:  [0.04546665 0.86263969 0.61561942 0.33840701]

###### - np.random.randint()
Genera números enteros aleatorios en rango y cantidad deseada

Estructura: np.random.randint(inicio, final, size=#)
- Inicio: Parametro más bajo del rango de números aleatorios
- Final Parametro más alto del rango de números aleatorios (no incluido)
- Size: Cantidad de números deseados
 
Entrada:

    print("5 números aleatorios entre 1 y 10:",np.random.randint(1, 10, size=5))

Salida: 

    5 números aleatorios entre 1 y 10: [5 6 6 6 8]

###### - np.random.normal()
Devuelve un  arreglo o una matriz de números generados al azar que siguen una distribución normal estándar (media 0, desviación estándar 1).

Estructura:    
  - np.random.randn(num)
num= cantidad de números generados

- np.random.randn(fil, col)
fil: número de filas de la matriz generada
col: número de columnas de la matriz generada

Entrada:

    print("4 números aleatorios con distribución normal estándar",np.random.randn(4))

Salida:

    4 números aleatorios con distribución normal estándar [-0.49199468 -0.8327575  -1.82536419  1.32795976]

###### - np.random.choice()
Selecciona elementos aleatorios de una lista o array.

Estructura:    
np.random.choice(lista, size=5) 
np.random.choice(lista, size=None, replace=True, p=None)

- lista= lista o array del que se toman los datos 
- size= cantidad de valores deseados a seleccionar en la lista (no obligatoria)
- replace= Si es True, permite repetir elementos, si es False, no se repiten(no obligatoria)
-p= Probabilidades asociadas a cada elemento. Debe ser un array del mismo tamaño que a, y sumar 1 (no obligatoria)

Entrada:

    opciones = ['profe', 'el', 'trabajo','esta', 'muy','largooo']
    probabilidades= [0.20,0.20,0.20,0.20,0.10,0.10]
    np.random.choice(opciones)              
    print("ejemplo:", np.random.choice(opciones, size=5, replace= True, p=probabilidades) )

Salida:

    ejemplo: ['el' 'esta' 'largooo' 'trabajo' 'profe']

###### - np.random.seed()
Establece una semilla para el generador de números aleatorios de NumPy. Esto significa que, si usas la misma semilla, obtendrás los mismos resultados aleatorios cada vez que ejecutes el código.

Estructura:    
np.random.seed(num)
- num= número que establece la semilla

Entrada:

    np.random.seed(42)
    print("A pesar de ser un randint en la semilla 42 este comando será 6 3 7 4 6: \n",np.random.randint(0, 10, size=5))

Salida:

    A pesar de ser un randint en la semilla 42 este comando será 6 3 7 4 6: 
    [6 3 7 4 6]


###### - np.random.uniform()
Genera números aleatorios uniformemente distribuidos en un rango personalizado.

Estructura:    
np.random.uniform(min, max, size=4)
- min: Parametro más bajo del rango de números aleatorios
- max Parametro más alto del rango de números aleatorios (no incluido)
- Size: Cantidad de números deseados 

Entrada:

    np.random.uniform(5, 10, size=4) 

Salida:

    array([5.07983126, 6.15446913, 6.20512733, 8.41631759])

###### - np.random.normal()
Genera números con una distribución normal personalizada.

Estructura:    
np.random.normal(loc=0.0, escala=1.0, size=None)
- loc: Media de la distribución deseada
- escala: desviación estandar de la distribución deseada
- Size: Cantidad de números deseados 

Entrada:

    np.random.normal(loc=10, scale=2, size=5)  # Media 10, desviación 2

Salida:

    array([12.26045639, 10.74623783,  9.2270541 ,  7.68245952, 11.13222565])

###### - np.random.shuffle()
Mezcla aleatoriamente los elementos de un arreglo.

Estructura:    
np.random.shuffle(arr)
- arr: Arreglo a mezclar

Entrada:

    arreglo = np.array([1, 2, 3, 4, 5])
    np.random.shuffle(arreglo)
    print("El orden ya no es 1,2,3,4,5 es:", arreglo)  # El array queda desordenado

Salida:

    El orden ya no es 1,2,3,4,5 es: [1 5 3 2 4]

Aclaración:

El módulo np.random incluye más de 90 funciones, muchas de ellas orientadas a aplicaciones avanzadas como simulaciones científicas, modelado estadístico complejo o control detallado del generador de números aleatorios. En este documento se han seleccionado únicamente los comandos más útiles y representativos para un nivel introductorio, priorizando aquellos que permiten generar datos simulados, trabajar con distribuciones comunes, mezclar arreglos y controlar la reproducibilidad. Estas funciones cubren los casos más frecuentes.
