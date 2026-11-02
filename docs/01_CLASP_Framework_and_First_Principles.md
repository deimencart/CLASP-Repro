---
category: [robotics, training, computer-vision]
status: validated
created: 2026-10-22
updated: 2026-10-22
source: "[[CLASP_SemanticKeypoints_2025]]"
tags: [clasp, foundation-models, semantic-keypoints, clothes-manipulation, affordance, first-principles, notebooklm-study]
---

# 01. CLASP: Fundamentos del Framework, Affordances Textiles y Primeros Principios

> **Fuente para Cuaderno Específico de NotebookLM: "CLASP & Foundation Models for Cloth Manipulation"**  
> Este documento desarrolla la deconstrucción conceptual y matemática del framework **CLASP (*Clothes Manipulation with Semantic Keypoints*)** (Deng, Tang, Yu, Li, Hsu — IJRR 2026 / ICRA 2025), explicando la superación del enfoque de reconstrucción 3D mediante anclajes semánticos canónicos.

---

## 0. Preámbulo Matemático: Geometría Diferencial de Superficies Monge y Operador de Forma (Fundamento Máster en Robótica / Percepción 3D)

Para transformar la detección de un píxel semántico 2D $(u^*, v^*)$ en un agarre robótico antipodal viable (ej. con la pinza Robotiq 2F-85), el robot debe comprender la **geometría diferencial local de la superficie deformable** a partir del mapa de profundidad métrico (*Sun et al., 2016; apuntes de máster [[Hybrid_Geometric_Grasp_Filtering]] y [[Sun_RobotVisionClothesManipulation_2016]]*).

### 0.1. Modelado de la Prenda como Parche Paramétrico de Monge
La imagen de profundidad local de la prenda se modela matemáticamente como una superficie diferenciable continua parametrizada en coordenadas cartesianas:
$$\mathbf{r}(x, y) = \begin{bmatrix} x \\ y \\ Z(x, y) \end{bmatrix} \in \mathbb{R}^3$$

Los vectores tangentes coordenados en el espacio de la cámara son:
$$\mathbf{r}_x = \frac{\partial \mathbf{r}}{\partial x} = \begin{bmatrix} 1 \\ 0 \\ Z_x \end{bmatrix}, \qquad \mathbf{r}_y = \frac{\partial \mathbf{r}}{\partial y} = \begin{bmatrix} 0 \\ 1 \\ Z_y \end{bmatrix}$$

El vector normal unitario exterior a la superficie del tejido se obtiene mediante el producto vectorial normalizado:
$$\mathbf{n}(x, y) = \frac{\mathbf{r}_x \times \mathbf{r}_y}{\|\mathbf{r}_x \times \mathbf{r}_y\|_2 + \epsilon} = \frac{1}{\sqrt{1 + Z_x^2 + Z_y^2} + \epsilon} \begin{bmatrix} -Z_x \\ -Z_y \\ 1 \end{bmatrix}, \qquad \epsilon = 10^{-7}$$

### 0.2. Primera y Segunda Forma Fundamental
1. **Primera Forma Fundamental (Métrica Intrínseca de Superficie):** Cuantifica cómo se miden longitudes y ángulos sobre la tela arrugada:
   $$\mathbf{I} = \begin{bmatrix} E & F \\ F & G \end{bmatrix} = \begin{bmatrix} \mathbf{r}_x \cdot \mathbf{r}_x & \mathbf{r}_x \cdot \mathbf{r}_y \\ \mathbf{r}_x \cdot \mathbf{r}_y & \mathbf{r}_y \cdot \mathbf{r}_y \end{bmatrix} = \begin{bmatrix} 1 + Z_x^2 & Z_x Z_y \\ Z_x Z_y & 1 + Z_y^2 \end{bmatrix}$$

2. **Segunda Forma Fundamental (Curvatura Extrínseca):** Cuantifica cómo se curva la tela respecto al espacio 3D ambiente:
   $$\mathbf{II} = \begin{bmatrix} L & M \\ M & N \end{bmatrix} = \begin{bmatrix} \mathbf{r}_{xx} \cdot \mathbf{n} & \mathbf{r}_{xy} \cdot \mathbf{n} \\ \mathbf{r}_{xy} \cdot \mathbf{n} & \mathbf{r}_{yy} \cdot \mathbf{n} \end{bmatrix} = \frac{1}{\sqrt{1 + Z_x^2 + Z_y^2} + \epsilon} \begin{bmatrix} Z_{xx} & Z_{xy} \\ Z_{xy} & Z_{yy} \end{bmatrix}$$

### 0.3. Operador de Weingarten (*Shape Operator*) y Curvaturas Principales
La matriz del Operador de Forma $\mathbf{W}$ mapea el espacio tangente en sí mismo:
$$\mathbf{W} = \mathbf{I}^{-1} \mathbf{II}$$

Los autovalores de $\mathbf{W}$ determinan las **curvaturas principales** extremas de la superficie $k_1$ y $k_2$ (ordenadas analíticamente como $k_1 \ge k_2$):
$$\det(\mathbf{W} - k \mathbf{I}_2) = 0 \implies k^2 - 2H k + K = 0$$
donde:
* $H = \frac{1}{2}(k_1 + k_2) = \frac{1}{2}\operatorname{tr}(\mathbf{W})$ es la **Curvatura Media**.
* $K = k_1 k_2 = \det(\mathbf{W})$ es la **Curvatura Gaussiana**.

### 0.4. Koenderink Shape Index ($S$) y Selección Físico-Geométrica de Agarre
Para desacoplar la forma local de la escala de la arruga, se utiliza el **Índice de Forma de Koenderink & van Doorn (1992)**:
$$S = \frac{2}{\pi} \arctan\left( \frac{k_1 + k_2}{k_1 - k_2} \right) \in [-1, +1], \qquad C = \sqrt{\frac{k_1^2 + k_2^2}{2}}$$

```
Índice de Forma S de Koenderink:
-1.0 ------- -0.625 -------- 0.0 ---------- +0.25 -------- +1.0
Copa Cóncava   Valle (Rut)    Silla/Paso      Cresta (Ridge) Cúpula Convexa
[PROHIBIDO]   [CONTACTO MESA] [Transición]   [AGARRE ÓPTIMO] [Esquina/Botón]
```

* **Cresta Convexa Agarrable (*Ridge*, $S \in [1/8, 3/8] = [0.125, 0.375]$):** La tela forma un pliegue levantado en el aire con espacio libre para que las dos mordazas de la pinza entren por debajo sin golpear la mesa.
* **Valle Cóncavo (*Rut*, $S \in [-7/8, -5/8] = [-0.875, -0.625]$):** La tela está presionada contra la mesa. Cualquier intento de agarre directo aquí colisionará las puntas de la pinza contra la mesa de trabajo.
* **Filtro de CLASP:** El punto semántico detectado por DINOv2 solo se valida para ejecución si su entorno cumple $S \in [1/8, 3/8]$ y curvatura mínima $C \ge C_{\min}$.

---

## 1. Deconstrucción en Primeros Principios: La Falacia de la Reconstrucción 3D en Ropa

En la robótica clásica de manipulación de objetos rígidos (cubos, botellas, herramientas mecánicas), el paradigma dominante ha sido **Reconstruir y Planificar**:
1. El robot escanea la escena con un sensor LiDAR o cámara RGB-D.
2. Un algoritmo geométrico (ICP, mallas de Poisson, Signed Distance Functions - SDF) reconstruye la geometría métrica 3D completa.
3. Se calcula una pose rígida en $SE(3)$ ($\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3$) y un planificador cinemático mueve la pinza al contacto.

### Por qué la Reconstrucción Falla Sistemáticamente en Prendas Textiles:
1. **Espacio de Configuración de Dimensión Infinita:** Un pantalón o camiseta es un medio continuo no rígido. Dos prendas idénticas pueden adoptar miles de millones de configuraciones arrugadas distintas que no admiten una transformación rígida $SE(3)$.
2. **Auto-Oclusiones Severas e Indiscernibles:** Los pliegues de la tela tapan capas interiores. Una cámara cenital fija solo observa la cara superior de un montón arrugado; no puede inferir si la tela debajo tiene 1, 2 o 5 pliegues superpuestos.
3. **Ausencia de Textura Distintiva:** Muchas prendas son monocromáticas (un vaquero azul, una camiseta negra), provocando que los descriptores geométricos clásicos de visión (SIFT, ORB) fallen por falta de gradientes visuales.

### El Cambio de Paradigma de CLASP: Affordances Semánticos Canónicos
Un ser humano que trabaja en una tintorería o dobla ropa en su casa **no reconstruye mentalmente una malla 3D de 50.000 polígonos**. Su sistema visual realiza un reconocimiento semántico de alto nivel:
* Identifica regiones anatómicas funcionales: *"la cintura"*, *"el dobladillo del tobillo izquierdo"*, *"el cuello"*, *"el puño de la manga"*.
* Sabe que estas regiones constituyen los **anclajes cinemáticos naturales (*affordances*)** para manipular la prenda, independientemente de cómo esté deformada.
* **Tesis de CLASP:** Si un robot es capaz de localizar de forma robusta estos anclajes semánticos en coordenadas métricas, la tarea de manipulación (doblar, colgar, alisar) se reduce a una secuencia cinemática ejecutable sobre dichos puntos, eliminando la necesidad de reconstruir la prenda completa o de entrenar políticas de RL de extremo a extremo (*pixel-to-torque*).

---

## 2. Arquitectura General de CLASP: Cero Entrenamiento y Encoders Congelados

Una de las contribuciones más radicales de CLASP es que **ninguna red neuronal es entrenada ni sometida a fine-tuning con backpropagation**. 

El framework se compone enteramente de **Modelos Fundacionales de Visión (*Vision Foundation Models - VFMs*) congelados (*frozen encoders*)** preentrenados a escala de Internet:

```
                            Imagen RGB-D de Entrada
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
        Segmentación (SAM)                        Vision-Language Model (VLM)
      Aísla máscara binaria                      Descubre categoría y propone
      de la prenda M_SAM                        Bounding Box grueso B_coarse
                |                                             |
                +----------------------+----------------------+
                                       |
                                       v
                     Fusión Espacio-Semántica Densa
               +-----------------------------------------------+
               | DINOv2 (ViT-B/14): Geometría local invariante  |
               | Stable Diffusion U-Net: Semántica de contexto |
               +-----------------------------------------------+
                                       |
                                       v
                   Mapa de Similitud Coseno (BBS Matching)
                   Frente al Prototipo Canónico Desplegado
                                       |
                                       v
                   Keypoint Semántico 2D (u*, v*) en Imagen
                                       |
                                       v
                     Desproyección Pinhole Z + Normal
                   Pose de Agarre 3D (X, Y, Z, theta_yaw)
```

---

## 3. Formalismo Matemático de Correspondencia Semántica

### 3.1. Prototipo Canónico por Categoría
Para cada categoría de ropa (pantalones, camisetas, faldas), CLASP requiere una **única imagen prototipo limpia** $I_{\text{proto}}$, tomada con la prenda completamente extendida y plana sobre una superficie. En este prototipo se anotan (manualmente o mediante VLM) los $K$ keypoints canónicos:

$$\mathcal{K}_{\text{proto}} = \big\{ \mathbf{p}_1^*, \mathbf{p}_2^*, \dots, \mathbf{p}_K^* \big\}, \qquad \mathbf{p}_k^* = (u_k, v_k) \in \mathbb{R}^2$$

### 3.2. Fusión de Descriptores de DINOv2 y Difusión Estable
Dada una imagen de consulta de la prenda arrugada $I_Q$, se extraen dos tensores de características complementarios:
1. **DINOv2 (ViT-B/14):** Descriptores de parches de salida de la última capa de atención:
   $$\mathbf{F}_{\text{DINO}} \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times D_1}, \qquad D_1 = 768$$
2. **Stable Diffusion:** Descriptores semánticos de alto nivel extraídos de los bloques intermedios del decodificador de la U-Net en el paso de difusión $t = 150$:
   $$\mathbf{F}_{\text{SD}} \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times D_2}, \qquad D_2 = 1280$$

Ambos mapas se interpolan bilinealmente a la resolución de imagen completa $(H, W)$ y se concatenan previa normalización euclidiana estricta con $\epsilon = 10^{-7}$:

$$\tilde{\mathbf{f}}(u, v) = \left[ \frac{\mathbf{f}_{\text{DINO}}(u, v)}{\|\mathbf{f}_{\text{DINO}}(u, v)\|_2 + \epsilon}, \; \frac{\mathbf{f}_{\text{SD}}(u, v)}{\|\mathbf{f}_{\text{SD}}(u, v)\|_2 + \epsilon} \right]^\top \in \mathbb{R}^{D_1 + D_2}$$

### 3.3. Similitud Coseno y Best-Buddies Similarity (BBS)
Para un keypoint canónico $\mathbf{p}_k^*$ con vector descriptor $\mathbf{f}_k^* \in \mathbb{R}^{D_1 + D_2}$ en el prototipo, el mapa de afinidad sobre la prenda arrugada se evalúa como:

$$S_k(u, v) = \frac{\tilde{\mathbf{f}}(u, v)^\top \mathbf{f}_k^*}{\|\tilde{\mathbf{f}}(u, v)\|_2 \|\mathbf{f}_k^*\|_2 + \epsilon}$$

Para prevenir falsos positivos causados por regiones simétricas (ej. emparejar el bolsillo derecho con el izquierdo), se impone el criterio **Best-Buddies Similarity (BBS)**: un píxel $p \in I_Q$ solo se considera candidato válido de $q \in I_{\text{proto}}$ si ambos son sus respectivos vecinos más cercanos mutuos en el espacio de embedding:

$$\text{BBS}(p, q) = \mathbb{I}\Big( q = \arg\max_{q'} \text{sim}(\mathbf{f}_p, \mathbf{f}_{q'}) \;\land\; p = \arg\max_{p'} \text{sim}(\mathbf{f}_q, \mathbf{f}_{p'}) \Big)$$

El punto semántico óptimo en 2D se obtiene combinando el mapa de afinidad con la máscara binaria de primer plano de SAM $\mathcal{M}_{\text{SAM}}$ y la ventana de contexto del VLM $\mathcal{B}_{\text{coarse}}$:

$$\mathbf{p}_k = \arg\max_{(u, v) \in \mathcal{B}_{\text{coarse}}} \Big( S_k(u, v) \cdot \mathcal{M}_{\text{SAM}}(u, v) \Big)$$

---

## 4. Síntesis de la Pose de Agarre en el Espacio Tridimensional

La localización 2D del píxel $(u^*, v^*)$ es insuficiente para ejecutar un agarre con una pinza de dos mordazas paralelas (ej. Robotiq 2F-85). Se requiere una pose tridimensional de agarre $\mathbf{G} = (\mathbf{P}_{\text{grip}}, \theta_{\text{yaw}}) \in \mathbb{R}^3 \times [-\pi, \pi]$.

### 4.1. Desproyección Métrica Pinhole
Utilizando el mapa de profundidad métrico alineado $Z = D(u^*, v^*)$ y la matriz de calibración intrínseca de la cámara $\mathbf{K}$:

$$\mathbf{P}_{\text{cam}} = \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = Z \cdot \mathbf{K}^{-1} \begin{bmatrix} u^* \\ v^* \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{(u^* - c_x) Z}{f_x} \\ \frac{(v^* - c_y) Z}{f_y} \\ Z \end{bmatrix}$$

### 4.2. Orientación del Vector de Ataque de las Mordazas ($\theta_{\text{yaw}}$)
Si la pinza desciende con sus mordazas orientadas paralelamente a la costura, colisionará lateralmente y empujará la tela fuera del agarre. Las mordazas deben alinearse de modo que su plano de cierre sea **estrictamente perpendicular a la frontera exterior del tejido**:

$$\mathbf{n}_{\text{edge}} = \nabla \mathcal{M}_{\text{SAM}}(u^*, v^*) = \begin{bmatrix} \frac{\partial \mathcal{M}}{\partial x} \\ \frac{\partial \mathcal{M}}{\partial y} \end{bmatrix}$$

El ángulo de yaw óptimo respecto al marco de la mesa resulta:

$$\theta_{\text{yaw}} = \operatorname{atan2}\big(n_y, n_x\big)$$

El comando de aproximación cartesiana proyecta las mordazas abiertas a una cota aérea $h_{\text{hover}} = 0.060\,\text{m}$ orientadas con $\theta_{\text{yaw}}$ antes de iniciar el descenso táctil.
