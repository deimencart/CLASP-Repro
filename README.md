# CLASP: Cloth Assistance with Semantic Keypoints & Interactive Perception

> **Reproducción Académica e Implementación Experimental**  
> *Referencia: CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints (Deng et al., 2025)*  
> *Tecnologías: PyTorch, DINOv2 ViT, Pinhole 3D, FSM de Control en Bucle Cerrado*

---

> [!IMPORTANT]
> **Aviso de Reproducción Académica (Sin fines de plagio)**  
> Este repositorio contiene una **reproducción experimental e implementación académica independiente** realizada con fines educativos y de investigación en robótica. Todos los conceptos teóricos, la formulación del problema y el diseño de la arquitectura original pertenecen y son atribuidos a sus autores originales (*Deng et al., 2025*). Esta implementación no tiene fines comerciales ni intención de plagio.

---

## 📌 ¿Qué problema resuelve este proyecto?

En la manipulación robótica de ropa, detectar dónde agarrar una prenda deformada (arrugada, doblada o desordenada) suele requerir entrenar detectores supervisados que fallan si cambia el color, el tipo de tela o la postura.

**CLASP** resuelve esto mediante dos ideas centrales:
1. **Transferencia Semántica Zero-Shot:** Usar un modelo de fundación visual congelado (**DINOv2**) que ya entiende partes semánticas de los objetos. Solo anotamos una prenda de referencia (prototipo canónico) y el algoritmo encuentra automáticamente esos mismos puntos (cintura, bolsillos, botamangas, tiro) en cualquier prenda nueva, sin reentrenar ninguna red neuronal.
2. **Percepción Interactiva en Bucle Cerrado:** La visión sola no basta porque la ropa resbala o se pueden agarrar múltiples capas por error. La pinza del robot utiliza contacto físico y lecturas hápticas para garantizar un agarre correcto antes de levantar la prenda.

---

## 🛠️ ¿Qué implementamos en este repositorio?

El pipeline implementado cubre desde la imagen RGB-D hasta la ejecución cinemática en el robot:

```
[ Prenda Prototipo Anotada ]           [ Prenda Real Observada (RGB-D) ]
             |                                       |
             +-------------------+-------------------+
                                 |
                                 v
                 [ Extractor DINOv2 ViT-B/14 ]
                   (Descriptores densos en GPU)
                                 |
                                 v
                 [ Matching Semántico Grueso a Fino ]
                   (Localización de keypoints)
                                 |
                                 v
                 [ Desproyección Pinhole 3D (SE3) ]
                   (Coordenadas cartesianas del robot)
                                 |
                                 v
                 [ FSM de Percepción Interactiva ]
                   1. Hover (+6 cm sobre la tela)
                   2. Touchdown táctil (Fz >= 3.0 N)
                   3. Verificación de grosor de mordazas
                   4. Elevación y verificación de agarre
```

### 1. Extracción y Emparejamiento Semántico
* **Extractor DINOv2 (`src/02_dino_extractor.py`):** Pasa la imagen por un ViT-B/14 congelado y normaliza los vectores descriptores por píxel en GPU.
* **Matcher de Keypoints (`src/03_matcher.py`):** Compara los descriptores de la prenda observada contra la prenda prototipo mediante similitud coseno, transfiriendo los puntos clave de agarre de forma inmediata.
* **Alineación de Rotación (`src/04_rotation_aligner.py`):** Normaliza la orientación 2D en el plano para facilitar el matching si la prenda está rotada.

### 2. Desproyección Pinhole a Marco del Robot
* **Desproyector 3D (`src/05_deprojector_3d.py`):** Toma las coordenadas de píxel $(u, v)$ y la profundidad métrica $Z$ de la cámara, desproyecta a coordenadas 3D y aplica la transformación mano-ojo $T_{\text{base}}^{\text{cam}}$ para darle la meta exacta en metros al brazo robótico.

### 3. Máquina de Estados de Agarre en Bucle Cerrado
* **Percepción Interactiva (`src/clasp_interactive_perception.py`):**
  * **Hover:** Posiciona el efector final a $+6\,\text{cm}$ sobre el punto de agarre en dirección normal a la superficie.
  * **Touchdown Háptico:** Desciende a velocidad controlada ($1.5\,\text{cm/s}$) hasta detectar un contacto vertical de $F_z \ge 3.0\,\text{N}$, evitando colisiones bruscas contra la mesa.
  * **Discriminación de Capas:** Mide la apertura residual de los dedos de la pinza para comprobar si atrapó tela con éxito (evitando cierres en falso o agarres de bordes dobles excesivamente gruesos).
  * **Lift & Verify:** Eleva la prenda $+15\,\text{cm}$ verificando la estabilidad de la carga.

---

## 📊 Evidencias y Resultados Visuales

El sistema transfiere los keypoints semánticos con precisión a diferentes prendas reales (vaqueros, chinos y pantalones de chándal):

| Vaqueros Reales (Prueba 2) | Vaqueros Reales (Prueba 3) |
| :---: | :---: |
| ![Matching Jeans 2](docs/images/match_real_jeans_2.png) | ![Matching Jeans 3](docs/images/match_real_jeans_3.png) |

| Transferencia a Pantalón Chino | Transferencia a Pantalón Deportivo |
| :---: | :---: |
| ![Matching Chino](docs/images/match_trousers_chino_beige.png) | ![Matching Sweatpants](docs/images/match_trousers_sweatpants_grey.png) |

---

## 📁 Estructura del Repositorio

```text
clasp_semantic_keypoints/
├── README.md                                 <- Documentación general
├── .gitignore                                <- Exclusiones de Git
├── docs/
│   ├── CLASP_SemanticKeypoints_2025.pdf      <- Paper científico original
│   ├── 01_CLASP_Framework_and_First_Principles.md
│   ├── 02_CLASP_Foundation_Models_Ecosystem.md
│   └── images/                               <- Evidencias de matching
├── src/
│   ├── 02_dino_extractor.py                  <- Extracción DINOv2 en GPU
│   ├── 03_matcher.py                         <- Algoritmo de correspondencia
│   ├── 04_rotation_aligner.py                <- Alineador de ángulo 2D
│   ├── 05_deprojector_3d.py                  <- Conversor Pinhole 2D a 3D robot
│   ├── clasp_interactive_perception.py       <- FSM de bucle cerrado (Touchdown + Agarre)
│   ├── action_primitives.py                  <- Primitivas de movimiento
│   ├── inference.py                          <- Script de evaluación completa
│   ├── simulation_utils.py                   <- Utilerías de simulación
│   └── metric.py                             <- Evaluación métrica de error
└── data/
    ├── prototypes/                           <- Prendas y anotaciones de referencia
    └── samples/                              <- Muestras fotográficas de prueba
```

---

## 🚀 Guía Rápida de Ejecución

### 1. Dependencias
```bash
pip install torch torchvision numpy opencv-python scipy matplotlib
```

### 2. Ejecutar Matching de Keypoints
Extrae descriptores y calcula la correspondencia semántica:
```bash
python src/02_dino_extractor.py
python src/03_matcher.py
```

### 3. Ejecutar Percepción Interactiva
Ejecuta la máquina de estados con aproximación, touchdown por fuerza y verificación de agarre:
```bash
python src/clasp_interactive_perception.py
```

---

## 📚 Referencias
* Deng, et al. (2025). *CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints*. arXiv:2502.17983.
* Oquab, M., et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. TMLR.
