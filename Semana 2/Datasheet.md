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

##### 7) Algebra lineal

###### - np.dot(A, B) / A @ B
Realiza el producto matricial entre dos arreglos o matrices.

Estructura: np.dot(A, B) A @ B

- A: Primer arreglo o matriz
- B: Segundo arreglo o matriz

Entrada:

    A = np.array([[1, 2], [3, 4]])  
    B = np.array([[5, 6], [7, 8]])  
    print("Producto matricial:\n", np.dot(A, B))

Salida: 

    Producto matricial:
    [[19 22]
    [43 50]]

###### - A.T
Devuelve la transpuesta de una matriz (intercambia filas por columnas).

Estructura: A.T
- A: Matriz a transponer

Entrada:
    
    A = np.array([[1, 2], [3, 4]])  
    print("Transpuesta:\n", A.T)

Salida: 

    Transpuesta:
    [[1 3]
    [2 4]]

###### - np.trace(A)
Calcula la traza de una matriz (suma de los elementos de la diagonal principal).

Estructura: np.trace(A)
- A: Matriz cuadrada

Entrada:

    A = np.array([[1, 2], [3, 4]])  
    print("Traza:", np.trace(A))

Salida: 

    Traza: 5 

###### - np.linalg.det(A)
Calcula el determinante de una matriz cuadrada.

Estructura: np.linalg.det(A)
- A: Matriz cuadrada

Entrada:

    A = np.array([[1, 2], [3, 4]])  
    print("Determinante:", np.linalg.det(A))

Salida: 

    Determinante: -2.0000000000000004

###### - np.linalg.inv(A)
Calcula la inversa de una matriz cuadrada.

Estructura: np.linalg.inv(A)
- A: Matriz cuadrada

Entrada:

    A = np.array([[1, 2], [3, 4]])  
    print("Inversa:\n", np.linalg.inv(A))

Salida: 

    Inversa:
    [[-2.   1. ]
    [ 1.5 -0.5]]

###### - np.linalg.matrix_rank(A)
Calcula el rango de una matriz (número de filas o columnas linealmente independientes).

Estructura: np.linalg.matrix_rank(A)
- A: Matriz

Entrada:

    A = np.array([[1, 2], [2, 4]])  
    print("Rango:", np.linalg.matrix_rank(A))

Salida: 

    Rango: 1

###### - np.linalg.norm(v)
Calcula la norma (magnitud o longitud) de un vector.

Estructura: np.linalg.norm(v)
- v: Vector

Entrada:

    v = np.array([3, 4])  
    print("Norma:", np.linalg.norm(v))

Salida: 

    Norma: 5.0

###### - np.linalg.solve(A, b)
Resuelve sistemas de ecuaciones lineales del tipo Ax = b.

Estructura: np.linalg.solve(A, b)

- A: Matriz de coeficientes
- b: Vector de resultados

Entrada:

    A = np.array([[3, 1], [1, 2]])  
    b = np.array([9, 8])  
    print("Solución:", np.linalg.solve(A, b))

Salida: 

    Solución: [2. 3.]

###### - np.linalg.eig(A)
Calcula los valores propios y vectores propios de una matriz cuadrada.

Estructura: np.linalg.eig(A)
- A: Matriz cuadrada

Entrada:

    A = np.array([[1, 2], [2, 1]])  
    valores, vectores = np.linalg.eig(A)  
    print("Valores propios:", valores)  
    print("Vectores propios:\n", vectores)

Salida: 

    Valores propios: [ 3. -1.]
    Vectores propios:
     [[ 0.70710678 -0.70710678]
     [ 0.70710678  0.70710678]]

###### - np.linalg.svd(A)
Realiza la descomposición en valores singulares (SVD), útil en compresión y machine learning.

Estructura: np.linalg.svd(A)
- A: Matriz

Entrada:

A = np.array([[1, 0], [0, -1]])  
U, S, V = np.linalg.svd(A)  
print("U:\n", U)  
print("S:", S)  
print("V:\n", V)

Salida: 

    U:
     [[1. 0.]
     [0. 1.]]

    S: [1. 1.]

    V:
     [[ 1.  0.]
     [-0. -1.]]

###### - np.identity(n)
Crea una matriz identidad de tamaño n×n.

Estructura: np.identity(n)

- n: Tamaño de la matriz

Entrada:

    print("Matriz identidad de 3x3:\n", np.identity(3))

Salida:

    Matriz identidad de 3x3:
     [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]

###### - np.diag(v)
Crea una matriz diagonal a partir de un vector, o extrae la diagonal de una matriz.

Estructura: np.diag(v)
- v: Vector o matriz

Entrada:

    v = np.array([1, 2, 3])  
    print("Matriz diagonal:\n", np.diag(v))

Salida: 

    Matriz diagonal:
     [[1 0 0]
     [0 2 0]
     [0 0 3]]

###### - np.cross(u, v)
Calcula el producto cruz entre dos vectores en 3D.

Estructura: np.cross(u, v)
- u: Primer vector
- v: Segundo vector

Entrada:

    u = np.array([1, 0, 0])  
    v = np.array([0, 1, 0])  
    print("Producto cruz:", np.cross(u, v))

Salida: 

    Producto cruz: [0 0 1]

###### Seccion C - Pandas y DataFrames
    -Rama: C-pandas-dataframes
        
        QUE ES PANDAS?
        
         El nombre Pandas es en realidad una contracción del término Panel Data para series de datos que incluyen observaciones a lo largo de varios periodos de tiempo. La biblioteca se creó como herramienta de alto nivel para el análisis en Python.Donde los creadores de Pandas pretenden que esta biblioteca evolucione hasta convertirse en la herramienta de análisis y manipulación de datos de código abierto más potente y flexible.
         
         Además del análisis de datos, Pandas se utiliza mucho para la Data Wrangling,permitiendo englobar los métodos de transformación de datos no estructurados para hacerlos procesables. Por lo general, Pandas también destaca en el procesamiento de datos estructurados en forma de tablas, matrices o series temporales.

         Pandas está diseñada específicamente para la manipulación y el análisis de datos en el lenguaje Python de esta manera logra ser potente, flexible y fácil de usar. Gracias a este por fin se puede utilizar el lenguaje Python para cargar, alinear, manipular o fusionar datos, donde el rendimiento es impresionante cuando el código fuente del back-end está escrito en C o Python.
         
        QUE ES UN DATAFRAME?

         Un DataFrame es una serie de Series Pandas indexadas por un valor, los DataFrames son paneles bidimensionales compuestos por filas y columnas, que permiten destacar las relaciones entre las distintas variables de la serie de datos a diferencia de las Series, que son objetos correspondientes a paneles unidimensionale.

        QUE ES UNA SERIE?

         Una series es la estructura de datos unidimensional etiquetada siendo similar a un array de 1D de NumPy, pero tiene un índice que permite el acceso a los valores por etiqueta donde serie puede contener cualquier tipo de datosn como enteros, cadenas, objetos de Python, etc.

         Una serie en Pandas tiene dos partes diferenciadas:
         -Indice (index): Que es un array de etiquetas asociado a los datos.
         -Valor (value): Por ende un value es un array de datos.

        QUE ES LA CARGA DE DATOS?
        
         La carga de datos significa importar información desde un archivo o fuente externa hacia un DataFrame para poder trabajar con ella en Python por eso es el primer paso en casi cualquier análisis de datos que consiste en traer la información cruda a un entorno manejable.

        QUE ES LA SELECCION DE COLUMNAS?

         Es el proceso de elegir una o varias columnas específicas de un DataFrame para analizarlas o manipularlas como por ejemplo: 
        
         Pensemos que un DataFrame es como una tabla en Excel: cada columna es un campo (nombre, edad, ciudad, etc.) y puedes trabajar solo con las que necesites, nombre-ciudad, nombre- edad, etc.

        Tipos de datos?

         En Padas, cada columna de un DataFrame o cada Serie tiene un tipo de dato, que indica la naturaleza de la información que contiene números, texto, fechas, etc.
         
         Esto es muy importante porque determina qué operaciones se pueden hacer, ayudando a optimizar la memoria y el rendimiento.
         
         QUE TIPOS DE DATOS?

         Numéricos 
         Enteros->int64, int32, int16.
         Decimales->float64, float32.
         
         Texto
         object->Usado para texto.
         string->Optimizado para trabajar con cadenas de caracteres.
         
         Booleanos
         bool->Valores verdaeros o falsos.
         
         Fechas y tiempos
         datetime64->Fechas y horas.
         timedelta->Diferencias de tiempo.
         
         Categorías
         category->Datos con un número limitado de valores repetidos. Muy útil para ahorrar memoria y acelerar cálculos.
        
        EJEMPLOS:
         
         DataFrame:
         
         //Paso 1: importar pandas
         import pandas as pd

         //Paso 2: crear un DataFrame sencillo
         data = {
            "Nombre": ["Ana", "Luis", "Marta"],
            "Edad": [23, 35, 29],
            "Ciudad": ["Bogotá", "Medellín", "Cali"]
         }

         df = pd.DataFrame(data)

         //Paso 3: mostrar el DataFrame
         print("DataFrame completo:")
         print(df)

         //Paso 4: ver información de los datos
         print("\n Tipos de datos de cada columna:")
         print(df.dtypes)

         // Paso 5: mostrar las primeras filas (head)
         print("\n Primeras filas del DataFrame:")
         print(df.head())

        Resultado:

          DataFrame completo:
          Nombre  Edad    Ciudad
        0    Ana    23    Bogotá
        1   Luis    35  Medellín
        2  Marta    29      Cali

          Tipos de datos de cada columna:
            Nombre    object
            Edad       int64
            Ciudad    object
            dtype: object

          Primeras filas del DataFrame:
            Nombre  Edad    Ciudad
          0    Ana    23    Bogotá
          1   Luis    35  Medellín
          2  Marta    29      Cali
        
        SERIES:
         
         import pandas as pd

         //Crear una Serie a partir de una lista
            serie = pd.Series([10, 20, 30, 40, 50])

            print(" Serie de ejemplo:")
            print(serie)

         //Acceder a elementos por índice
            print("\n Elemento en la posición 0:", serie[0])
            print(" Elemento en la posición 3:", serie[3])

         //Crear una Serie con etiquetas personalizadas (índice)
            serie_con_etiquetas = pd.Series(
            [100, 200, 300],
            index=["A", "B", "C"]
            )

            print("\n Serie con etiquetas personalizadas:")
            print(serie_con_etiquetas)

         //Acceder usando la etiqueta
            print("\n Valor con etiqueta 'B':", serie_con_etiquetas["B"])
        
        Resultado:

        Serie de ejemplo:
            0    10
            1    20
            2    30
            3    40
            4    50
         dtype: int64

         Elemento en la posición 0: 10
         Elemento en la posición 3: 40

        Serie con etiquetas personalizadas:
            A    100
            B    200
            C    300
         dtype: int64

        Valor con etiqueta 'B': 200

        Carga de datos:

            import pandas as pd

            //Paso 1: Crear un archivo CSV como ejemplo
            data = """Nombre,Edad,Ciudad
            Ana,23,Bogotá
            Luis,35,Medellín
            Marta,29,Cali
            """
            with open("personas.csv", "w") as f:
            f.write(data)

            //Paso 2: Cargar el CSV con Pandas
            df = pd.read_csv("personas.csv")

            //Paso 3: Mostrar resultados
            print(" DataFrame cargado desde CSV:")
            print(df)

            print("\n Tipos de datos:")
            print(df.dtypes)

            print("\n Primeras filas:")
            print(df.head())

        Resultado:

            DataFrame cargado desde CSV:
            Nombre  Edad    Ciudad
          0    Ana    23    Bogotá
          1   Luis    35  Medellín
          2  Marta    29      Cali

            Tipos de datos:
            Nombre    object
            Edad       int64
            Ciudad    object
            dtype: object

            Primeras filas:
            Nombre  Edad    Ciudad
          0    Ana    23    Bogotá
          1   Luis    35  Medellín
          2  Marta    29      Cali

        Seleccion de columnas:

            import pandas as pd

            # Crear un DataFrame de ejemplo
            data = {
            "Nombre": ["Ana", "Luis", "Marta"],
            "Edad": [23, 35, 29],
            "Ciudad": ["Bogotá", "Medellín", "Cali"]
            }

            df = pd.DataFrame(data)

            print(" DataFrame original:")
            print(df)

            # --- Selección de UNA columna ---
            print("\n Selección de la columna 'Nombre':")
            print(df["Nombre"])   # Devuelve una Serie

            # --- Selección de VARIAS columnas ---
            print("\n Selección de 'Nombre' y 'Ciudad':")
            print(df[["Nombre", "Ciudad"]])   # Devuelve un DataFrame

            # --- Selección usando notación de punto (solo si no hay espacios en el nombre) ---
            print("\n Selección de la columna 'Edad' con notación de punto:")
            print(df.Edad)

        Resultado:

         DataFrame original:
          Nombre  Edad    Ciudad
        0    Ana    23    Bogotá
        1   Luis    35  Medellín
        2  Marta    29      Cali

          Selección de la columna 'Nombre':
         0      Ana
         1     Luis
         2    Marta
         Name: Nombre, dtype: object

          Selección de 'Nombre' y 'Ciudad':
          Nombre    Ciudad
        0    Ana    Bogotá
        1   Luis  Medellín
        2  Marta      Cali

          Selección de la columna 'Edad' con notación de punto:
            0    23
            1    35
            2    29
          Name: Edad, dtype: int64

        Tipos de datos:

            import pandas as pd

            //Crear un DataFrame con varios tipos de datos
            data = {  
            "Nombre": ["Ana", "Luis", "Marta"],             # Texto (object)
            "Edad": [23, 35, 29],                           # Enteros (int64)
            "Ingreso": [1500.50, 2500.00, 1800.75],         # Decimales (float64)
            "Activo": [True, False, True],                  # Booleanos (bool)
            "Fecha_ingreso": pd.to_datetime(["2023-05-10", "2022-11-20", "2024-01-15"]), # Fecha (datetime64)
            "Ciudad": pd.Categorical(["Bogotá", "Medellín", "Cali"]) # Categórico
            }

            df = pd.DataFrame(data)

            print(" DataFrame:")
            print(df)

            print("\n Tipos de datos:")
            print(df.dtypes)

        Resultado:

         DataFrame:
         Nombre  Edad  Ingreso  Activo Fecha_ingreso    Ciudad
       0    Ana    23  1500.50    True    2023-05-10    Bogotá
       1   Luis    35  2500.00   False    2022-11-20  Medellín
       2  Marta    29  1800.75    True    2024-01-15      Cali

         Tipos de datos:
         Nombre                   object
         Edad                      int64
         Ingreso                 float64
         Activo                     bool
         Fecha_ingreso    datetime64[ns]
         Ciudad                 category
         dtype: object