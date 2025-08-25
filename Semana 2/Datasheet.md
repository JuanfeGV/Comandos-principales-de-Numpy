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

