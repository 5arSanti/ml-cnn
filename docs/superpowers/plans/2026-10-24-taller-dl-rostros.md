# Taller DL Rostros Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completar el taller oficial Olivetti (MLP, SVM+PCA, CNN) y añadir el punto 6 que reconoce las 10 fotos propias como persona `Yo` (etiqueta 40).

**Architecture:** Se copia la plantilla a `docs/Taller_DL_Rostros.ipynb` y no se toca el original. Los puntos 1–5 usan la partición oficial 256/64/80. El punto 6 recorta `images/` con Haar, parte 6/2/2, concatena copias de esos splits y reentrena los tres modelos con 41 clases.

**Tech Stack:** Python 3, NumPy, scikit-learn, TensorFlow/Keras, OpenCV, Matplotlib, pandas, seaborn, Jupyter.

## Global Constraints

- Plantilla `docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb`: no modificar.
- Entrega: `docs/Taller_DL_Rostros.ipynb` (único archivo de código).
- No commitear `images/` (fotos personales, ~14 MB).
- Semilla: `tf.keras.utils.set_random_seed(42)` y `random_state=42`.
- Partición oficial: no rebarajar ni rehacer `train_test_split` de Olivetti.
- Extra: etiqueta `40`, nombre visible `Yo`; Haar `scaleFactor=1.1`, `minNeighbors=5`; pad 20%; fallback recorte central.
- CNN oficial: solo cambio (a) data augmentation. Dropout 0.5. `Dense(40)`. Extra: misma CNN con `Dense(41)`.
- SVM: `C ∈ {1, 10}`, `gamma ∈ {"scale", 0.01}`, sin `probability=True`.
- Evidencia = notebook ejecutado (sin pytest), como dice el spec.
- Comentarios orales en español junto a scaler, PCA+whiten, overfit y aplanado.
- Bitácora PDF: fuera de alcance (se llena a mano).

## File structure

| Path | Responsibility |
|---|---|
| `docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb` | Plantilla. No tocar. |
| `docs/Taller_DL_Rostros.ipynb` | Notebook de entrega: puntos 1–5 + punto 6. |
| `.gitignore` | Ignorar `images/`, checkpoints, `__pycache__/`. |
| `images/` | 10 fotos de entrada. Solo lectura. |

No crear módulos `.py` aparte: el enunciado y el spec piden un notebook que corra en Colab.

---

### Task 1: Andamiaje del notebook de entrega

**Files:**
- Create: `docs/Taller_DL_Rostros.ipynb` (copia de la plantilla)
- Create: `.gitignore`
- Do not modify: `docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb`

**Interfaces:**
- Consumes: plantilla con celdas 0–16 (datos oficiales ya resueltos, TODOs en MLP/SVM/CNN).
- Produces: notebook de trabajo listo para rellenar TODOs; `.gitignore` que excluye `images/`.

- [ ] **Step 1: Crear `.gitignore`**

```gitignore
images/
.ipynb_checkpoints/
__pycache__/
*.pyc
```

- [ ] **Step 2: Copiar la plantilla**

```bash
cp docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb docs/Taller_DL_Rostros.ipynb
```

Expected: el archivo nuevo existe y `diff` contra la plantilla está vacío.

- [ ] **Step 3: Añadir una línea al markdown del título (primera celda)**

Al final de la primera celda markdown, agregar:

```markdown

Copia de trabajo (la plantilla `02_Taller_DL_Rostros_ESTUDIANTES.ipynb` no se modifica).
Puntos 1–5 = enunciado oficial. Punto 6 = extra: reconocer fotos propias (`Yo`, etiqueta 40).
```

No cambiar ninguna celda de carga ni de partición (`faces = fetch_olivetti_faces...` ni los dos `train_test_split`).

- [ ] **Step 4: Comprobar que la plantilla original no cambió**

```bash
diff docs/02_Taller_DL_Rostros_ESTUDIANTES.ipynb docs/Taller_DL_Rostros.ipynb
```

Expected: solo el extra de 3 líneas en la primera celda. Si `diff` muestra cambios en celdas de datos, revertir esos cambios.

- [ ] **Step 5: Commit**

```bash
git add .gitignore docs/Taller_DL_Rostros.ipynb
git commit -m "feat: scaffold the workshop notebook from the student template"
```

---

### Task 2: MLP oficial

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celda del TODO de MLP, la que importa `MLPClassifier`)

**Interfaces:**
- Consumes: `Xtr_flat`, `Xte_flat`, `y_train`, `y_test`, `resultados` de las celdas oficiales.
- Produces: `mlp`, `acc_mlp`, `resultados["MLP"]` (float).

- [ ] **Step 1: Reemplazar la celda TODO del MLP por este código**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# El scaler se ajusta SOLO con entrenamiento: si usáramos prueba,
# filtraríamos información que el modelo no debería haber visto.
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr_flat)
Xte_s = scaler.transform(Xte_flat)  # transform, no fit

mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
mlp.fit(Xtr_s, y_train)

acc_mlp = accuracy_score(y_test, mlp.predict(Xte_s))
resultados["MLP"] = acc_mlp
print(f"MLP — accuracy en prueba: {acc_mlp:.3f}")
```

- [ ] **Step 2: Ejecutar las celdas 0–8 (imports, datos, MLP)**

En Jupyter / Colab, o:

```bash
jupyter nbconvert --to notebook --execute docs/Taller_DL_Rostros.ipynb --ExecutePreprocessor.timeout=600 --output /tmp/taller_mlp.ipynb
```

Si `nbconvert` no está, abrir el notebook y correr hasta el MLP.

Expected:
- `Entrenamiento: (256, 64, 64, 1) | Validación: (64, 64, 64, 1) | Prueba: (80, 64, 64, 1)`
- `MLP — accuracy en prueba: 0.XXX` con `XXX` típicamente ≥ 0.80
- `resultados["MLP"]` definido

- [ ] **Step 3: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: train official MLP on flattened Olivetti faces"
```

---

### Task 3: SVM + PCA oficial

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celda TODO de SVM + PCA)

**Interfaces:**
- Consumes: `X_temp`, `Xte_flat`, `y_temp`, `y_test`, `pipe` ya declarado en la plantilla, `resultados`.
- Produces: `grid`, `acc_svm`, `resultados["SVM + PCA"]`, print de `grid.best_params_`.

- [ ] **Step 1: Reemplazar la celda TODO de SVM por este código**

Conservar el `Pipeline` que ya está en esa celda. Sustituir solo los `...`:

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 4096 vars y 256 fotos: PCA a 100 eigenfaces. whiten=True iguala la
# varianza de cada componente para que el RBF no lo domine el primer eje.
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100, whiten=True, random_state=42)),
    ("svm", SVC(kernel="rbf")),
])

param_grid = {
    "svm__C": [1, 10],
    "svm__gamma": ["scale", 0.01],
}

# GridSearch usa train+val (X_temp): el CV interno hace las particiones.
Xtemp_flat = X_temp.reshape(len(X_temp), -1)
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(Xtemp_flat, y_temp)

print("Mejores parámetros:", grid.best_params_)
acc_svm = accuracy_score(y_test, grid.predict(Xte_flat))
resultados["SVM + PCA"] = acc_svm
print(f"SVM + PCA — accuracy en prueba: {acc_svm:.3f}")
```

- [ ] **Step 2: Ejecutar esa celda (tras el MLP)**

Expected:
- Print `Mejores parámetros: {'svm__C': ..., 'svm__gamma': ...}`
- `SVM + PCA — accuracy en prueba: 0.XXX` típicamente ≥ 0.85
- `resultados` tiene `MLP` y `SVM + PCA`

- [ ] **Step 3: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: add official SVM+PCA with GridSearchCV"
```

---

### Task 4: CNN oficial con data augmentation

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celda del `Sequential` y la celda de evaluación)

**Interfaces:**
- Consumes: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, `graficar_historia`.
- Produces: `model`, `history`, `acc_cnn`, `resultados["CNN"]`. Una sola llamada a `model.evaluate` en prueba.

- [ ] **Step 1: Reemplazar la celda de la CNN por este código**

```python
model = keras.Sequential([
    keras.Input(shape=(64, 64, 1)),
    # (a) data augmentation: más variedad con solo 256 fotos
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),

    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(40, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, start_from_epoch=15, restore_best_weights=True
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[early],
)

# Si accuracy train >> val y val_loss sube, hay sobreajuste.
graficar_historia(history, "— mi CNN")
```

- [ ] **Step 2: Reemplazar la celda de evaluación CNN por este código**

```python
# Una sola evaluación en prueba (el enunciado lo pide explícitamente).
_, acc_cnn = model.evaluate(X_test, y_test, verbose=0)
resultados["CNN"] = float(acc_cnn)
print(f"CNN — accuracy en prueba: {acc_cnn:.3f}")
print("Cambio(s) realizado(s): data augmentation (flip horizontal, rotación 0.05, zoom 0.1)")
```

No añadir tercer `Conv2D`. No cambiar `Dropout(0.5)`. No volver a llamar `evaluate` en un bucle.

- [ ] **Step 3: Ejecutar las dos celdas**

Expected:
- `model.summary()` con última capa `dense` de 40 unidades.
- Curvas accuracy/loss.
- `CNN — accuracy en prueba: 0.XXX` típicamente ≥ 0.80
- El print del cambio (a).

En CPU puede tardar varios minutos; en GPU T4, 1–3 minutos. Misma semilla, mismo código.

- [ ] **Step 4: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: train official CNN with data augmentation"
```

---

### Task 5: Tabla comparativa y borrador de reflexión

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celda de la tabla ya existe; celda markdown de reflexión)

**Interfaces:**
- Consumes: `resultados` con keys `"MLP"`, `"SVM + PCA"`, `"CNN"`; `y_test` de longitud 80.
- Produces: DataFrame ordenado; párrafo editable en la celda 5.

- [ ] **Step 1: Dejar la celda de la tabla como en la plantilla (ya está completa)**

```python
import pandas as pd
tabla = pd.DataFrame({"Accuracy (prueba)": resultados}).sort_values("Accuracy (prueba)", ascending=False)
tabla["Errores (de 80)"] = ((1 - tabla["Accuracy (prueba)"]) * len(y_test)).round().astype(int)
tabla
```

No cambiar la fórmula. Tras ejecutar, las tres filas deben existir.

- [ ] **Step 2: Añadir una celda de verificación oficial justo después de la tabla**

```python
assert X_train.shape == (256, 64, 64, 1), X_train.shape
assert X_val.shape == (64, 64, 64, 1), X_val.shape
assert X_test.shape == (80, 64, 64, 1), X_test.shape
assert set(resultados) == {"MLP", "SVM + PCA", "CNN"}, set(resultados)
counts = np.bincount(y_test, minlength=40)
assert (counts == 2).all(), counts
print("Verificación oficial OK:", {k: round(v, 3) for k, v in resultados.items()})
```

Expected al correrla: `Verificación oficial OK: {...}` y ningún `AssertionError`.

- [ ] **Step 3: Sustituir el placeholder de reflexión por este borrador editable**

En la celda markdown del punto 5, debajo de `**Respuesta:**`, poner:

```markdown
**Respuesta (borrador — editar con nuestras palabras antes de la socialización):**

En este problema de *imágenes*, el modelo que más alto quedó en la tabla de prueba fue el primero de `tabla` (escribir el nombre y el accuracy al terminar de correr el notebook). Tiene sentido que una red densa no sea la mejor herramienta: al aplanar cada foto a 4096 píxeles, el MLP trata cada pixel como una variable suelta y pierde el hecho de que un ojo es un *vecindario* de píxeles. SVM+PCA primero resume la foto en 100 eigenfaces (direcciones de máxima varianza) y el kernel RBF separa personas en ese espacio chico; la CNN, además, mira parches locales con las convoluciones. En el taller de crédito los datos eran *tabulares* (ingresos, mora, etc.): ahí una regresión logística suele bastar y una red no aporta estructura espacial que no existe. En mi interés profesional usaría SVM cuando el dataset es mediano, las variables ya están bien definidas y necesito un modelo estable y explicable por márgenes, no cuando el input es una imagen cruda.
```

Las tres preguntas del enunciado deben seguir visibles encima del párrafo.

- [ ] **Step 4: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: add official comparison table and reflection draft"
```

---

### Task 6: Punto 6 — rutas, Haar y grilla de recortes

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celdas nuevas al final)

**Interfaces:**
- Consumes: carpeta `images/` (10 archivos `.jpg`/`.jpeg`/`.png`).
- Produces: `ETIQUETA_YO = 40`, `NOMBRE_YO = "Yo"`, `resolve_images_dir()`, `list_image_files()`, `crop_face_64()`, `yo_paths`, `yo_X` shape `(10, 64, 64, 1)`, grilla original vs recorte.

- [ ] **Step 1: Añadir markdown de sección**

```markdown
## 6. Extra: ¿nos reconoce a nosotros?
Fuera de la rúbrica. Concatenamos 10 fotos propias (etiqueta 40, nombre **Yo**) a *copias* de la partición oficial. Olivetti no se rebaraja.
```

- [ ] **Step 2: Añadir celda de dependencias y funciones**

```python
from pathlib import Path
import cv2

ETIQUETA_YO = 40
NOMBRE_YO = "Yo"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def list_image_files(folder):
    return sorted(
        [p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name,
    )


def resolve_images_dir():
    for raw in ("images", "../images", "/content/images"):
        p = Path(raw)
        if p.is_dir() and list_image_files(p):
            return p.resolve()
    return None


def _center_square(gray):
    h, w = gray.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return gray[y0:y0 + side, x0:x0 + side]


def crop_face_64(path):
    """Devuelve (recorte 64x64x1 en [0, 1], aviso o None)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    cascade = cv2.CascadeClassifier(HAAR_XML)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    aviso = None
    if len(faces) == 0:
        recorte = _center_square(gray)
        aviso = f"Haar no detectó cara en {path.name}; usé recorte central."
    else:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad_x, pad_y = int(0.20 * w), int(0.20 * h)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x + w + pad_x)
        y1 = min(gray.shape[0], y + h + pad_y)
        recorte = gray[y0:y1, x0:x1]
    recorte = cv2.resize(recorte, (64, 64)).astype("float32") / 255.0
    return recorte[..., np.newaxis], aviso
```

- [ ] **Step 3: Añadir celda que carga, valida el conteo y dibuja la grilla**

```python
images_dir = resolve_images_dir()
if images_dir is None:
    raise FileNotFoundError(
        "No encontré images/. En local: carpeta images/ del repo. "
        "En Colab: sube las 10 fotos a /content/images y vuelve a correr esta celda."
    )

yo_paths = list_image_files(images_dir)
if len(yo_paths) != 10:
    raise RuntimeError(
        f"El extra exige exactamente 10 fotos; hallé {len(yo_paths)} en {images_dir}."
    )

yo_X = []
avisos = []
originales = []
for p in yo_paths:
    bgr = cv2.imread(str(p))
    originales.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    crop, aviso = crop_face_64(p)
    yo_X.append(crop)
    if aviso:
        avisos.append(aviso)
        print(aviso)

yo_X = np.stack(yo_X, axis=0)
print("yo_X:", yo_X.shape, "rango", yo_X.min(), yo_X.max())

fig, axes = plt.subplots(10, 2, figsize=(6, 22))
for i, p in enumerate(yo_paths):
    axes[i, 0].imshow(originales[i])
    axes[i, 0].set_title(p.name[:28], fontsize=8)
    axes[i, 0].axis("off")
    axes[i, 1].imshow(yo_X[i, :, :, 0], cmap="gray")
    axes[i, 1].set_title("recorte 64×64", fontsize=8)
    axes[i, 1].axis("off")
plt.tight_layout()
plt.show()

assert yo_X.shape == (10, 64, 64, 1)
assert 0.0 <= float(yo_X.min()) and float(yo_X.max()) <= 1.0
```

- [ ] **Step 4: Ejecutar las celdas nuevas desde el repo root (cwd donde exista `images/` o `../images`)**

Si el kernel arranca en `docs/`, `../images` debe resolver. Expected:
- `yo_X: (10, 64, 64, 1)`
- Grilla 10×2
- A lo sumo avisos Haar (el notebook no debe caerse)
- Si falta la carpeta: el `FileNotFoundError` con el texto de Colab, no un crash opaco

- [ ] **Step 5: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: crop personal photos to Olivetti-sized faces"
```

---

### Task 7: Punto 6 — split 6/2/2 y concatenación

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celda nueva)

**Interfaces:**
- Consumes: `yo_X`, `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` (oficiales, sin mutar).
- Produces: `X_train_41`, `X_val_41`, `X_test_41`, `y_train_41`, `y_val_41`, `y_test_41`, flats `Xtr_flat_41`, `Xval_flat_41`, `Xte_flat_41`, `Xtemp_flat_41`, `y_temp_41`, e índices/máscara de las 2 fotos `Yo` en prueba.

- [ ] **Step 1: Añadir esta celda**

```python
# Split propio 6/2/2. Una sola clase: no hay stratify.
# random_state=42 + yo_paths ya ordenados por nombre = partición fija.
yo_y = np.full(len(yo_X), ETIQUETA_YO, dtype=y_train.dtype)
idx = np.arange(len(yo_X))

idx_temp, idx_test = train_test_split(idx, test_size=0.20, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.25, random_state=42)

yo_train, yo_val, yo_test = yo_X[idx_train], yo_X[idx_val], yo_X[idx_test]
yo_y_train = yo_y[idx_train]
yo_y_val = yo_y[idx_val]
yo_y_test = yo_y[idx_test]
print("Fotos Yo — train/val/test:", len(yo_train), len(yo_val), len(yo_test))
print("Held-out Yo:", [yo_paths[i].name for i in idx_test])

# Copias concatenadas: Olivetti no se rebaraja (la tabla de 40 clases sigue siendo la del enunciado).
X_train_41 = np.concatenate([X_train, yo_train], axis=0)
X_val_41 = np.concatenate([X_val, yo_val], axis=0)
X_test_41 = np.concatenate([X_test, yo_test], axis=0)
y_train_41 = np.concatenate([y_train, yo_y_train], axis=0)
y_val_41 = np.concatenate([y_val, yo_y_val], axis=0)
y_test_41 = np.concatenate([y_test, yo_y_test], axis=0)

Xtr_flat_41 = X_train_41.reshape(len(X_train_41), -1)
Xval_flat_41 = X_val_41.reshape(len(X_val_41), -1)
Xte_flat_41 = X_test_41.reshape(len(X_test_41), -1)
Xtemp_flat_41 = np.concatenate([Xtr_flat_41, Xval_flat_41], axis=0)
y_temp_41 = np.concatenate([y_train_41, y_val_41], axis=0)

assert X_train.shape == (256, 64, 64, 1)
assert X_train_41.shape == (262, 64, 64, 1)
assert X_val_41.shape == (66, 64, 64, 1)
assert X_test_41.shape == (82, 64, 64, 1)
assert (y_test_41[-2:] == ETIQUETA_YO).all()
print("Shapes 41:", X_train_41.shape, X_val_41.shape, X_test_41.shape)
```

- [ ] **Step 2: Ejecutar la celda**

Expected:
```
Fotos Yo — train/val/test: 6 2 2
Shapes 41: (262, 64, 64, 1) (66, 64, 64, 1) (82, 64, 64, 1)
```

`X_train.shape` sigue `(256, 64, 64, 1)`.

- [ ] **Step 3: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: append person-41 photos to copied Olivetti splits"
```

---

### Task 8: Punto 6 — reentrenar los tres modelos y mostrar held-out

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (celdas nuevas)

**Interfaces:**
- Consumes: `*_41`, `yo_test`, `idx_test`, `yo_paths`, mismos hiperparámetros del oficial.
- Produces: `resultados_41` con accuracy en 82 fotos y aciertos `Yo` (0–2); figura de las 2 fotos held-out; frase si algún modelo falla.

- [ ] **Step 1: Celda MLP extra**

```python
resultados_41 = {}
aciertos_yo = {}

scaler_41 = StandardScaler()
Xtr_s_41 = scaler_41.fit_transform(Xtr_flat_41)
Xte_s_41 = scaler_41.transform(Xte_flat_41)

mlp_41 = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
mlp_41.fit(Xtr_s_41, y_train_41)
pred_mlp_41 = mlp_41.predict(Xte_s_41)
resultados_41["MLP"] = accuracy_score(y_test_41, pred_mlp_41)
aciertos_yo["MLP"] = int((pred_mlp_41[-2:] == ETIQUETA_YO).sum())
print(f"MLP 41 — acc prueba: {resultados_41['MLP']:.3f} | Yo held-out: {aciertos_yo['MLP']}/2")
```

- [ ] **Step 2: Celda SVM extra**

```python
pipe_41 = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=100, whiten=True, random_state=42)),
    ("svm", SVC(kernel="rbf")),
])
param_grid_41 = {
    "svm__C": [1, 10],
    "svm__gamma": ["scale", 0.01],
}
grid_41 = GridSearchCV(pipe_41, param_grid_41, cv=3)
grid_41.fit(Xtemp_flat_41, y_temp_41)
print("Mejores parámetros (41):", grid_41.best_params_)
pred_svm_41 = grid_41.predict(Xte_flat_41)
resultados_41["SVM + PCA"] = accuracy_score(y_test_41, pred_svm_41)
aciertos_yo["SVM + PCA"] = int((pred_svm_41[-2:] == ETIQUETA_YO).sum())
print(f"SVM + PCA 41 — acc prueba: {resultados_41['SVM + PCA']:.3f} | Yo held-out: {aciertos_yo['SVM + PCA']}/2")
```

- [ ] **Step 3: Celda CNN extra**

```python
model_41 = keras.Sequential([
    keras.Input(shape=(64, 64, 1)),
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.Conv2D(32, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(41, activation="softmax"),
])
model_41.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_41 = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, start_from_epoch=15, restore_best_weights=True
)
history_41 = model_41.fit(
    X_train_41, y_train_41,
    validation_data=(X_val_41, y_val_41),
    epochs=100,
    callbacks=[early_41],
)
graficar_historia(history_41, "— CNN 41 clases")
_, acc_cnn_41 = model_41.evaluate(X_test_41, y_test_41, verbose=0)
proba_cnn_41 = model_41.predict(X_test_41, verbose=0)
pred_cnn_41 = proba_cnn_41.argmax(axis=1)
resultados_41["CNN"] = float(acc_cnn_41)
aciertos_yo["CNN"] = int((pred_cnn_41[-2:] == ETIQUETA_YO).sum())
print(f"CNN 41 — acc prueba: {resultados_41['CNN']:.3f} | Yo held-out: {aciertos_yo['CNN']}/2")
```

- [ ] **Step 4: Celda tabla extra + figura held-out + criterio de éxito**

```python
def etiqueta_legible(k):
    return NOMBRE_YO if int(k) == ETIQUETA_YO else f"Persona {int(k)}"


tabla_41 = pd.DataFrame({
    "Accuracy (82 fotos)": resultados_41,
    "Aciertos Yo (de 2)": aciertos_yo,
}).sort_values("Accuracy (82 fotos)", ascending=False)
display(tabla_41)

# MLP y CNN: probabilidad de la clase predicha. SVM: solo la clase (sin probability=True).
proba_mlp_yo = mlp_41.predict_proba(Xte_s_41[-2:])
pred_mlp_yo = pred_mlp_41[-2:]
pred_svm_yo = pred_svm_41[-2:]
pred_cnn_yo = pred_cnn_41[-2:]
proba_cnn_yo = proba_cnn_41[-2:]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i in range(2):
    axes[i].imshow(X_test_41[-2 + i, :, :, 0], cmap="gray")
    titulo = (
        f"{yo_paths[idx_test[i]].name[:22]}\n"
        f"MLP: {etiqueta_legible(pred_mlp_yo[i])} ({proba_mlp_yo[i].max():.2f})\n"
        f"SVM: {etiqueta_legible(pred_svm_yo[i])}\n"
        f"CNN: {etiqueta_legible(pred_cnn_yo[i])} ({proba_cnn_yo[i].max():.2f})"
    )
    axes[i].set_title(titulo, fontsize=9)
    axes[i].axis("off")
plt.tight_layout()
plt.show()

if max(aciertos_yo.values()) < 2:
    print(
        "Ningún modelo acertó las 2 fotos Yo. Causa probable: 6 fotos de train, "
        "dominio celular vs Olivetti, o el selfie (otra luz / pose). "
        "No reentrenamos para forzar el acierto."
    )
elif min(aciertos_yo.values()) < 2:
    print("Al menos un modelo reconoció las 2 fotos Yo; otro no. Dejamos el resultado.")
else:
    print("Los tres modelos clasificaron las 2 fotos held-out como Yo.")
```

Éxito del extra: `max(aciertos_yo.values()) >= 2`. Si no, **no** retocar hiperparámetros ni repetir el fit.

- [ ] **Step 5: Ejecutar las celdas del extra**

Expected:
- Tres prints `... Yo held-out: n/2`
- Tabla de 3 filas
- Figura con 2 recortes
- Una de las tres frases de cierre

- [ ] **Step 6: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "feat: retrain three models to recognize personal photos"
```

---

### Task 9: Celda final de verificación

**Files:**
- Modify: `docs/Taller_DL_Rostros.ipynb` (última celda)

**Interfaces:**
- Consumes: todos los arrays y diccionarios de las tasks 2–8.
- Produces: print `Verificación extra OK` o `AssertionError` concreto.

- [ ] **Step 1: Añadir esta celda al final**

```python
assert X_train.shape == (256, 64, 64, 1)
assert X_val.shape == (64, 64, 64, 1)
assert X_test.shape == (80, 64, 64, 1)
assert X_train_41.shape == (262, 64, 64, 1)
assert X_val_41.shape == (66, 64, 64, 1)
assert X_test_41.shape == (82, 64, 64, 1)
assert (np.bincount(y_test, minlength=40) == 2).all()
assert set(resultados) == {"MLP", "SVM + PCA", "CNN"}
assert set(resultados_41) == {"MLP", "SVM + PCA", "CNN"}
assert set(aciertos_yo) == {"MLP", "SVM + PCA", "CNN"}
assert all(v in (0, 1, 2) for v in aciertos_yo.values())
assert yo_X.shape == (10, 64, 64, 1)
print("Verificación extra OK")
print("Oficial:", {k: round(float(v), 3) for k, v in resultados.items()})
print("41 clases:", {k: round(float(v), 3) for k, v in resultados_41.items()})
print("Aciertos Yo:", aciertos_yo)
```

- [ ] **Step 2: Correr la celda**

Expected: `Verificación extra OK` y los tres prints. Si falla un assert, corregir el bug de la task que produjo ese array; no silenciar el assert.

- [ ] **Step 3: Commit**

```bash
git add docs/Taller_DL_Rostros.ipynb
git commit -m "test: add end-of-notebook shape and result assertions"
```

---

## Self-review (plan vs spec)

| Spec | Task |
|---|---|
| Plantilla intacta, copia de entrega | 1 |
| MLP scaler-only-train, (256,), max_iter=500 | 2 |
| SVM Pipeline + GridSearch C/gamma + X_temp | 3 |
| CNN solo augmentation, EarlyStopping, evaluate una vez | 4 |
| Tabla errores de 80 + reflexión 3 preguntas | 5 |
| Haar, pad 20%, fallback central, grilla, 10 fotos exactas | 6 |
| 6/2/2 seed 42, concat, shapes 262/66/82 | 7 |
| Tres modelos extra, tabla 82 + aciertos Yo, SVM sin proba | 8 |
| Criterio de éxito / no forzar retrain | 8 step 4 |
| Verificación shapes, keys, 2 por clase, held-out | 5 step 2 y 9 |
| Rutas images / Colab /content/images | 6 |
| No commitear images/ | 1 `.gitignore` |
| Comentarios orales | 2, 3, 4, 7 |
| Fuera de alcance (MediaPipe, bitácora, etc.) | no hay tasks para eso |
