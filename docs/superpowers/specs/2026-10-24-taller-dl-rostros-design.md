# Taller DL Rostros — diseño

Fecha: 2026-10-24  
Curso: Machine Learning · Universidad Libre · Gustavo Santos  
Fuente: `docs/Taller_DL_Rostros_Enunciado_y_Bitacora.pdf` y plantilla `docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb`

## Problema

Comparar tres modelos del curso (MLP, SVM+PCA, CNN) en clasificación de 40 personas (Olivetti Faces), con la misma partición. Además, en una sección extra que no altera esa partición, incluir las 10 fotos propias de `images/` como una persona más y mostrar que los tres modelos pueden reconocerlas.

## Decisiones cerradas

- Un notebook de entrega, dos experimentos encadenados (oficial 40 clases, extra 41 clases).
- Plantilla de estudiantes intacta; el trabajo vive en una copia.
- Fotos propias: detección Haar, recorte, grises, 64×64; las 10 fotos entran (incluye el selfie con gato).
- Partición propia 6 / 2 / 2 con `random_state=42`, concatenada a los splits oficiales (Olivetti no se rebaraja).
- Reentrenar los tres modelos en el extra, con los mismos hiperparámetros.
- Cambio de CNN del taller: solo data augmentation (opción a del enunciado).
- Nombre visible de la clase extra: `Yo`. Etiqueta numérica: `40`.

## Entregable

| Archivo | Rol |
|---|---|
| `docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb` | Plantilla. No modificar. |
| `docs/Taller_DL_Rostros.ipynb` | Notebook que se completa, ejecuta y entrega. |
| `images/` | 10 fotos propias. No forman parte de la rúbrica oficial. |

El notebook debe ejecutarse en Google Colab (GPU T4, como pide el enunciado) y en local. La carpeta de fotos es la **primera** de estas rutas que exista y contenga archivos imagen:

1. `images` (cwd)
2. `../images` (si el cwd es `docs/`)
3. `/content/images` (Colab)

Si ninguna sirve, mostrar cómo crear `/content/images` (o `images/` local) y no ejecutar el punto 6.

Semilla global: `tf.keras.utils.set_random_seed(42)` y `random_state=42` en sklearn.

Comentarios cortos en español junto a las decisiones que el profesor suele preguntar (scaler solo en train, PCA+whiten, overfit en curvas, por qué aplanar pierde localidad).

La bitácora en papel del PDF se llena a mano. Queda fuera de este trabajo.

## Experimento oficial (puntos 1–5)

No modificar la carga ni la partición.

1. `fetch_olivetti_faces(shuffle=True, random_state=42)`.
2. `images` shape `(400, 64, 64)`, 40 clases, 10 fotos cada una, píxeles ya en `[0, 1]`.
3. `X = images[..., np.newaxis]` → `(400, 64, 64, 1)`.
4. `train_test_split` estratificado: 20% prueba, del resto 20% validación. Resultado: `256 / 64 / 80`.
5. Flatten: `(n, 4096)` para MLP y SVM.

### 1. MLP (1.0 de la rúbrica)

- `StandardScaler` ajustado **solo** con `Xtr_flat`.
- `MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)`.
- `resultados["MLP"] = accuracy_score(y_test, pred)`.

### 2. SVM + PCA (1.0)

- Pipeline: `StandardScaler` → `PCA(n_components=100, whiten=True, random_state=42)` → `SVC(kernel="rbf")`.
- `GridSearchCV(cv=3)` sobre `svm__C ∈ {1, 10}` y `svm__gamma ∈ {"scale", 0.01}`.
- Ajuste sobre train+val aplanado (`X_temp`), como indica la plantilla.
- Imprimir mejores parámetros. `resultados["SVM + PCA"]` = accuracy en prueba.

### 3. CNN (1.5)

Un cambio respecto a la demo: data augmentation al inicio.

```
Input (64, 64, 1)
RandomFlip("horizontal")
RandomRotation(0.05)
RandomZoom(0.1)
Conv2D(32, 3, relu) + MaxPooling2D(2)
Conv2D(64, 3, relu) + MaxPooling2D(2)
Flatten
Dense(128, relu)
Dropout(0.5)
Dense(40, softmax)
```

- `optimizer="adam"`, `loss="sparse_categorical_crossentropy"`, `metrics=["accuracy"]`.
- `EarlyStopping(monitor="val_loss", patience=15, start_from_epoch=15, restore_best_weights=True)`.
- `epochs=100`, `validation_data=(X_val, y_val)`.
- Graficar accuracy y pérdida (entrenamiento vs validación) con `graficar_historia`.
- Evaluar **una sola vez** en prueba. `resultados["CNN"]` = esa accuracy.
- Texto explícito: `Cambio(s) realizado(s): data augmentation (flip horizontal, rotación 0.05, zoom 0.1)`.

No se añade un tercer bloque convolucional ni se cambia el Dropout. Un solo cambio, el (a).

### 4. Tabla (0.5)

DataFrame con accuracy en prueba y `Errores (de 80) = round((1 - acc) * 80)`, ordenado de mayor a menor accuracy.

### 5. Reflexión (1.0)

Tres preguntas del enunciado, en un párrafo que el estudiante pueda decir en voz alta:

1. ¿Qué modelo ganó aquí y por qué? (una red densa aplana y pierde vecindad espacial.)
2. ¿Qué cambia frente al problema de crédito, donde ganó la regresión logística?
3. ¿En qué escenario profesional usarían SVM en lugar de una red neuronal?

El notebook deja las preguntas y un **borrador editable**. No es texto para leerle al profesor.

## Experimento extra (punto 6)

No mueve `X_train`, `X_val`, `X_test` del oficial. Construye copias concatenadas.

### Preproceso de `images/`

Las 10 fotos (`.jpg` / `.jpeg` / `.png`), ordenadas por nombre, incluido `6f9effca-IMG_20250912_090120.jpg` (selfie con gato).

Por cada archivo:

1. `cv2.imread(..., cv2.IMREAD_GRAYSCALE)`.
2. `CascadeClassifier` Haar frontal de OpenCV (`haarcascade_frontalface_default.xml`), `detectMultiScale` con `scaleFactor=1.1` y `minNeighbors=5`.
3. Si hay varios recuadros, usar el de **mayor área** (la cara humana; el gato no dispara este cascade).
4. Expandir el recuadro 20% en cada lado, recortado a los bordes de la imagen, para no cortar frente ni barbilla.
5. Redimensionar a 64×64, float32 en `[0, 1]`, shape `(64, 64, 1)`.
6. Si Haar no detecta nada: recorte central cuadrado + `print` de aviso con el nombre del archivo. No se aborta el notebook.

Antes de entrenar, mostrar una grilla original vs recorte (10 pares) para inspección visual.

### Partición propia

`train_test_split` con `random_state=42`, sin stratify (una sola clase):

1. 20% prueba → 2 fotos.
2. Del resto, 25% validación → 2 fotos, 6 de entrenamiento.

Etiqueta `40`. Nombre para títulos: `Yo`.

### Unión

```
X_train_41 = concat(X_train, yo_train)     # 262
X_val_41   = concat(X_val,   yo_val)       # 66
X_test_41  = concat(X_test,  yo_test)      # 82
```

Flatten paralelo `(n, 4096)`. Olivetti no se vuelve a barajar. El orden de las fotos propias queda fijado por `sorted` de nombres de archivo + el split con semilla 42.

### Modelos extra

Mismos hiperparámetros que el oficial.

| Modelo | Diferencia vs oficial |
|---|---|
| MLP | Scaler reajustado en `X_train_41` aplanado. |
| SVM + PCA | GridSearch sobre train+val de 41 clases. Misma grilla. |
| CNN | Misma augmentation y EarlyStopping. Última capa `Dense(41, softmax)`. |

No se reutilizan pesos del experimento oficial. Se entrenan desde cero.

### Salidas del extra

1. Tabla: accuracy en las 82 fotos de prueba **y** aciertos en las 2 fotos `Yo` held-out (0, 1 o 2), para los tres modelos.
2. Figura: esas 2 fotos con, por modelo, etiqueta predicha y confianza (probabilidad de la clase predicha; en SVM, `decision_function` no es probabilidad — reportar la clase predicha y, si `probability=True` no está activo, no inventar un %). SVM se deja **sin** `probability=True` (evita un fit extra). Mostrar solo la clase predicha para SVM.
3. Criterio de éxito: las 2 fotos de prueba clasificadas como `Yo` en **al menos un** modelo. Objetivo deseable: los tres. Si falla, no reentrenar hasta acertar; dejar el resultado y una frase de causa probable (pocas muestras, dominio celular vs Olivetti, selfie distinto).

## Manejo de errores

| Situación | Comportamiento |
|---|---|
| Haar no encuentra cara | Recorte central cuadrado + aviso. El notebook sigue. |
| Falta `images/` | Mensaje: usar `images/` del repo o, en Colab, subir las 10 fotos a una carpeta `images`. No ejecutar el punto 6 sin esa carpeta. |
| La carpeta no tiene exactamente 10 archivos `.jpg`/`.jpeg`/`.png` | Abortar el extra con el conteo hallado. No improvisar otro split. |
| Sin GPU | Mismo código en CPU. Más lento. |
| Fallo de reconocimiento en held-out | Mostrar el error. No iterar hiperparámetros para “arreglarlo”. |

## Verificación (sin pytest)

El notebook corrido es la evidencia. Al final del oficial y del extra, celdas que comprueban:

1. Shapes oficiales: train/val/test = `(256, 64, 64, 1)` / `(64, …)` / `(80, …)`.
2. Shapes extra: `(262, …)` / `(66, …)` / `(82, …)`.
3. `y_test` oficial tiene exactamente 2 fotos por cada clase `0..39`.
4. `resultados` tiene las keys `MLP`, `SVM + PCA`, `CNN`.
5. La CNN oficial se evaluó una vez en prueba (una celda, no un bucle de reintentos).
6. Grilla de 10 recortes visible.
7. Predicción impresa de las 2 fotos held-out.

## Fuera de alcance

- MediaPipe, MTCNN u otro detector.
- Fine-tuning o redes preentrenadas (VGG-Face, FaceNet, etc.).
- Fotos de una segunda persona / pareja.
- App, API o UI.
- Modificar la partición oficial.
- Bitácora PDF (mano).
- Commit de las fotos personales al repo (privacidad; pesan ~14 MB).

## Control oral — puntos a poder explicar

- Por qué `StandardScaler` no se ajusta con validación ni prueba.
- Por qué 4096 variables y 256 fotos piden PCA, y qué es `whiten=True`.
- Qué se ve en las curvas si hay sobreajuste (accuracy train alta, val baja; loss val que sube).
- Qué pierde el MLP al aplanar: un píxel no “sabe” que su vecino es mejilla o fondo.
- Por qué el extra concatena y no rehace el `train_test_split` de Olivetti: la tabla de 40 clases debe seguir siendo comparable y defendible ante el enunciado.

## Flujo de datos

```
Olivetti 400 → split oficial 256/64/80 → MLP / SVM+PCA / CNN(40)
                                              ↓
                                         tabla + reflexión

images/ 10 → gris → Haar (o centro) → 64×64
         → split 6/2/2 etiqueta 40
         → concat a copias del split oficial
         → MLP / SVM+PCA / CNN(41)
         → tabla extra + 2 fotos held-out
```
