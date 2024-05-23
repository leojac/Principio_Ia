# Principio_IA

Código creado en un programa de maestría en IA, excluyendo la investigación o el análisis realizado en él.

## 1. Procesamiento de lenguaje

En esta sección se verán diversos manejos con corpus, como tratar los datos

### 1.1 Manejo de datos por letra

Analizo el package movie_reviews de NLTK, sacando los caracteres más usados, como las palabras más usadas, como contestar preguntas en base de los datos.

***librerías usadas:***
> **NLTK | Pandas | Itertools | Matplotlib**

### 1.2. Manejo de imágenes

Manejo una imagen con OpenCv donde se escaló en grises y escala de colores, de partir de tenerla se practicó extrayendo por píxeles y aplicación del filtro gaussiano con un kernel de 3x3 como otros filtros (nítida, erosión, morfológico, filtrado, grad)

***librerías usadas:***
> **Opencv | NumPy | Sklearn | Matplotlib | scipy**

### 1.3. Algoritmo de RANSAC

Búsqueda de puntos de intereses SIFT girando y mapeando las dos imágenes, por último se aplicó el algoritmo RANSAC para la trasformación de las dos imágenes

***librerías usadas:***
> **Opencv | NumPy | Sklearn | Matplotlib | scipy**

### 1.4 Detector de caras y blur

Se crea una función que nos permita capturar imágenes en un video con opencv posterior se crea otra donde trabajaremos con la distorsión del entorno en la imagen de dos formas  por medio de colores y de planos con su flujo y su cálculo para ver el desplazamiento de objetos.
Se realizó una función donde se nos permita localizar las caras de las personas y se aplica una función que nos permite difuminar el rostro de las personas.

***librerías usadas:***
> **Matplotlib | NumPy | Opencv | IPython | pillow | sys | base64 | google.colab | video | common**

### 1.5. Opencv y matplotlib

Conversión de imágenes a sus colores primarios, como sus escalas de grises y desplazamientos o rotación

***librerías usadas:***
> **Opencv | Matplotlib | NumPy**

### 1.6. Red neuronal Bert y GPT

Se genera un dataframe con la información del corpus load_wine donde se saca la frecuencia de los datos, posteriormente se entrenan los datos y se crea su matriz confusión, se le aplica el modelo DecisionTreeClassifier para esos datos pre-entrenados para realizar el trasform bert y GPt-2 a texto.

***librerías usadas:***
> **Opencv | Matplotlib | NumPy | nltk | pandas | seaborn | sklearn | plotly | transformers**

### 1.7. Hidden Markov Model Tagger

Se define la función de explora_tagset_fd para trabajar con los parámetros en el etiquetado del corpus [brown], después se procede a usar la función nltk.tag.hmm.HiddenMarkovModelTagger (symbols, states, transitions, outputs, priors, transform) con los valores definidos.
Se declara la función de entrenamientoHMM para el uso del entrenamiento del corpus trabajado y de entrenar el modelo HiddenMArkovModelTagger con el método train.

***librerías usadas:***
> **nltk | pprint | plotly | string | pandas**

### 1.8. Arquitectura Pix2Pix (batchnorm y dropout)

Se usará la librería para el acceso al drive propio, (se tendrá que extraer él .rar de los archivos y guardarlo en carpetas). Se crearán las capas de batchnorm y dropout y se crea una función donde se meterá 4 filtros.
Se procede a hacer una función comparativa de pérdida y se implementa un optimizador de modelo y, por último, se crea la función final donde sé entrena y demostrarán las imágenes generadas.

***librerías usadas:***
> **google.colab | tensorflow | Matplotlib | NumPy**

### 1.9. Palabras en imágenes

Se descarga un package de nltk donde se hará un diccionario con 3 campos que nos ayudaran a hacer una comparación de conjunto de palabras que vaya encontrando en el campo tweet y así discerniendo los valores importantes Probamos con dos clasificadores para poder sacar el más óptimo de los dos [naive bayes y SVM] 
Posterior Graficamos por medio de la matriz de confusión para poder apreciar más los falsos positivos o verdaderos positivos y se dividieron por palabras y se fueron eliminando las repetidas y al mismo tiempo contabilizándolas para sacar su repetitividad plasmándolo en una imagen.

***librerías usadas:***
> **nltk | plotly | pandas | collections | sklearn | Matplotlib | seaborn | pillow | NumPy | wordcloud**

## 2. Aprendizaje automático

En esta sección se estuvo trabajando con varios métodos y modelos, como trabajar con gratificación y matrices.

### 2.1. Ecuaciones lineales

un repaso por las maneras más básicas para resolver una ecuación lineal con los métodos *[método de ecuación lineal(solve numpy), la Regla de Cramer, Método de Jacobi, Método de gauss seldel, Eliminación de Gauss]* junto a como calcular la inversa de una matriz 4x4 y su traspuesto, multiplicarlo por escala y realizar el producto interno con otra matriz

***librerías usadas:***
> **NumPy**

### 2.2. Vectores

un repaso del uso de los vectores en 2 dimensiones y en 3 dimensiones, cómo sacar su producto, punto, longitud, ángulo, valor condenado y proyecciones.

***librerías usadas:***
> **Matplotlib | NumPy**

### 2.3. Derivadas

Se realizaron diversas derivadas (la primera derivada y segunda derivada de una función)

***librerías usadas:***
> **Matplotlib | NumPy | pandas | sympy**

### 2.4 propagación inversa

se creó con los datos iris del dataset de sklearn un algoritmo backpropagation (propagación inversa) donde se consiguieron buenos resultados en el modelo

***librerías usadas:***
> **Matplotlib | NumPy | pandas | sklearn**

### 2.5. Visualizacion de funciones

Se visualizó una función para ver la facilidad de creación de Matplotlib como al encontrar su mínimo valor estimado

***librerías usadas:***
> **Matplotlib | NumPy | sympy**

### 2.6. Conjunto Datos

Se realiza el análisis de discriminarte lineal Python y utiliza este conocimiento para llevar a cabo la reducción de la dimensión y la representación gráfica del primer conjunto de datos construido donde se vea la división en grupos (Hiperplano y Maximal Margin Classifier) en los datos, como una pequeña prueba de funcionamiento de las métricas del error cuadrático medio en scikit-learn

***librerías usadas:***
> **Matplotlib | NumPy | sklearn | pandas | plotly**

### 2.7. Matriz confusión

Se creó un árbol de decisiones con base en datos registrados. Posteriormente de entrenar el modelo de regresión logística, se hizo la matriz confusión.

***librerías usadas:***
> **Matplotlib | NumPy | sklearn | pandas | seaborn | IPython | pydotplus**

### 2.8. Matriz corrosiva

Utilizando el conjunto de datos de la Flor de Iris (que se encuentra integrado dentro de la librería Scikit-learn), elabora un programa en Python que compare los resultados de realizar la clasificación multinomial, a partir de un modelo discriminante (regresión logística) y un modelo generativo (Naive Bayes). Incluye tus conclusiones en una presentación digital

***librerías usadas:***
> **xgboost | NumPy | sklearn | pandas | seaborn | Matplotlib**

### 2.9. Programación genética

Se realizó el problema de la mochila por medio de la programación genética

***librerías usadas:***
> **random | NumPy**

### 2.10. Clasificación y agrupación

Con el database de iris se realiza el agrupamiento de dichos datos, utilizando el algoritmo k-means. En esta ocasión utiliza el agrupamiento jerárquico aglomerativo, genera el dendrograma correspondiente utilizando el método de linkage completo y realiza el análisis de los índices silhouette para seleccionar el número óptimo de clústeres.

***librerías usadas:***
> **SciPy | NumPy | sklearn | pandas | Matplotlib**

### 2.11. Clasificación  y agrupamiento de audio

    Pendiente

***librerías usadas:***
> **IPython | minisom | librosa | pandas | NumPy | Matplotlib | sklearn**

### 2.12. Reductor  de peso en las imagenes

se elaboraron las bases con las que se hace la reducción en una imagen de peso (junto a una alternativa más simple y actual).

***librerías usadas:***
> **NumPy | pandas | Matplotlib | imageio | math**

### 2.13. Ruta óptima

Se creó desde una base de datos para el entrenamiento para hacer una simulación, buscando la ruta más efectiva de un problema de ejemplo.

|  Ejemplo_1:               |      Ejemplo_2:         |
|:-------------------------:|:-----------------------:|
|Embalaje: 1                |Embalaje: 1              |
|Depósito: 1                |Depósito: 1              |
|Ancho(cm): 15,0            |Ancho(cm): 15,0          |
|Largo(cm) 40,0             |Largo(cm) 40,0           |
|Alto(cm): 20,0             |Alto(cm): 20,0           |
|Procedencia: A             |Procedencia: A           |
|Manipulación: Normal       |Manipulación: Normal     |
|Temperatura: refrigerado   |Temperatura: refrigerado |
|Protocolo: Protocolo_1     |Protocolo: Protocolo_1   |

Considera lo siguiente

- cada producto que se encuentra en el almacén central
- se identifica por el tipo de embalaje
- también posee como propiedad sus dimensiones (largo, ancho y altura con un máximo de 100,0 cm y un mínimo de 5,0 cm en cada dimensión)
- peso (kg),
- el tipo de manipulación que requiere (frágil, normal)
- lugar de procedencia (A, B, C, D)
- temperatura de almacenamiento (ambiente, refrigerado).
- tiene integrados dos protocolos de manipulación de carga (Protocolo_1 y Protocolo_2)

realiza las siguientes operaciones:

1. Genera un conjunto de datos de prueba (mínimo 10,000 muestras), considerando los criterios antes indicados y teniendo en cuenta tu experiencia personal, donde, a partir del análisis de la información del producto, estos puedan ser catalogados dentro los protocolos disponibles.
2. La planta industrial tiene una distribución como la que se muestra la imagen

![Imagen de idea del problema en esquema](../main/recursos/Img/image.png)

- Cada vehículo puede transportar un máximo de tres productos al mismo tiempo.
- Los productos en el almacén no poseen un orden determinado.
- Un vehículo puede seleccionar cualquier producto con la misma probabilidad.

> [!IMPORTANT]
> Nunca repita dos veces el mismo destino en cada viaje.
> Los productos del protocolo_1 tengan prioridad.
> La ruta trazada sea siempre la menor.

***librerías usadas:***
> **pandas | NumPy | sklearn | SciPy | Matplotlib**

## 3. Aprendizaje profundo

### 3.1. McCulloch-Pitts

Se hizo el modelo de neurona McCulloch-Pitts que debe recibir como parámetros los valores del conjunto de entradas, los pesos y un valor de umbral de activación y regresa la tabla de verdad de la neurona.

***librerías usadas:***
> **NumPy**

### 3.2. Modelo de neurona simple

se realizó dos funciones sigmoide y la ReLU en el conjunto de datos iris desde el datasets de sklearn

***librerías usadas:***
> **pandas | NumPy | sklearn | Matplotlib**

### 3.3. Modelo estocastico y adam

    Pendiente

***librerías usadas:***
> **sklearn**

### 3.4. MLP

se realizó el modelo MLP en el conjunto de MNIST y se mostró el rendimiento por X número de iteraciones

***librerías usadas:***
> **NumPy | sklearn | Matplotlib**

### 3.5. Dropout

    Pendiente

### 3.6. Arquitectura redes neuronales

se creó una arquitectura simple de redes neuronales con la función sigmoide

***librerías usadas:***
> **NumPy**

## 4. inteligencia artificial

### 4.1. Detector de objetos

Se creó un detector de objetos con datos locales (este caso se entrenó con 4000 imágenes clasificadas, se pueden agregar si se desea que detecte otro tipo) donde se usó TensorFlow con Keras para el modelo con 50 épocas y realizando su gráfica de precisión y pérdida con los datos de entrenamiento y pruebas, teniendo un buen desempeño

***librerías usadas:***
> **os | Matplotlib | tensorflow | NumPy | Pillow | requests | io | IPython | opencv**

### 4.2. Ibm Watson Pipeline

Se elaboró una API donde se consuma un modelo creado en IBM (Watson AutoAI), y se hicieron sus métricas de evaluación

***librerías usadas:***
> **ibm_watson_machine_learning**
