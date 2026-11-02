---
category: [computer-vision, foundation-models, clasp]
status: validated
created: 2026-10-22
updated: 2026-10-22
source: "[[01_CLASP_Framework_and_First_Principles]]"
tags: [dinov2, sam, stable-diffusion, owl-vit, foundation-models, vision-transformers, notebooklm-study]
---

# 02. Ecosistema de Modelos Fundacionales de Visión en CLASP: DINOv2, SAM y Difusión

> **Fuente para Cuaderno Específico de NotebookLM: "CLASP & Foundation Models for Cloth Manipulation"**  
> Este documento analiza en profundidad los cuatro modelos fundacionales que componen el pipeline de **CLASP**, detallando por qué su combinación resuelve la percepción de ropa deformable sin requerir ningún reentrenamiento.

---

## 1. DINOv2: Descriptores Espaciales con Invarianza Geométrica

### 1.1. Origen y Naturaleza Arquitectónica
**DINOv2** (Oquab et al., Meta AI 2024) es un Vision Transformer (ViT) entrenado de forma completamente auto-supervisada sobre un corpus de más de 142 millones de imágenes no etiquetadas (*LVD-142M*). Utiliza una combinación de destilación de conocimiento auto-supervisada (DINO) con modelado de parches enmascarados (*Masked Image Modeling - iBOT*).

### 1.2. Por qué DINOv2 es Crucial para Ropa Deformable
A diferencia de los modelos supervisados clásicos (ej. ResNet entrenado en ImageNet), DINOv2 no descarta la información geométrica y textural local para predecir una clase global. Sus características presentan propiedades únicas:
1. **Correspondencia Semántica de Parches:** Si extraemos el token de atención de una esquina de una mesa o del dobladillo de un pantalón, ese vector es extremadamente similar al del mismo componente en otra prenda de distinto color o textura.
2. **Robustez ante Deformaciones No Rígidas:** Los tokens preservan la topología local incluso si la tela está doblada a $45^\circ$ o arrugada.
3. **Invarianza a Condiciones de Iluminación y Sombras:** Las sombras arrojadas por los pliegues de la tela no confunden los embeddings de parches.

### 1.3. Extracción de Descriptores en PyTorch
```python (genesis_env)
import torch
import torchvision.transforms as T

# Carga del modelo fundacional congelado desde Torch Hub
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').cuda()
dinov2_vitb14.eval()

# Inferencia congelada sin autograd para ahorrar memoria VRAM
with torch.inference_mode():
    # Imagen de entrada normalizada: (1, 3, H, W) con H, W múltiplos de 14
    features_dict = dinov2_vitb14.forward_features(image_tensor)
    # Patch tokens densos: shape (1, (H/14)*(W/14), 768)
    patch_tokens = features_dict['x_norm_patchtokens']
```

---

## 2. Stable Diffusion: Descriptores de Semántica Global de Alto Nivel

### 2.1. El Rol de los Modelos Generativos en la Percepción Visual
Aunque Stable Diffusion (Rombach et al., CVPR 2022) fue concebido como un modelo generativo de texto a imagen, la investigación reciente (ej. DIFT - Diffusion Features, Tang et al. 2023) descubrió que **las capas intermedias de su U-Net actúan como los descriptores semánticos más potentes descubiertos hasta la fecha**.

### 2.2. Complementariedad DINOv2 vs Stable Diffusion
* **DINOv2 solo:** Es extraordinariamente preciso en bordes finos y geometría local, pero a veces sufre de ambigüedad semántica global (ej. puede confundir la costura de la cintura con la costura de una pernera porque localmente ambas son líneas de hilo idénticas).
* **Stable Diffusion solo:** Comprende perfectamente la estructura contextual global ("esto es la parte superior del torso", "esto es la pierna"), pero sus mapas de activación son de baja resolución espacial y tienen bordes borrosos.
* **La Fusión en CLASP:** Al concatenar $\mathbf{F}_{\text{DINO}} \oplus \mathbf{F}_{\text{SD}}$, CLASP obtiene lo mejor de ambos mundos: la precisión geométrica milimétrica de DINOv2 guiada por la semántica conceptual infalible de Stable Diffusion.

---

## 3. Segment Anything Model (SAM): Máscaras de Aislamiento Zero-Shot

### 3.1. Arquitectura de Tres Etapas de SAM
**SAM** (Kirillov et al., Meta AI 2023) se divide en:
1. **Image Encoder (Pesado):** Un Vision Transformer MAE que procesa la imagen una sola vez y genera un embedding denso de la escena.
2. **Prompt Encoder (Ultra-rápido):** Procesa puntos, cajas delimitadoras o máscaras previas en fracciones de milisegundo.
3. **Lightweight Mask Decoder:** Un decodificador basado en transformadores que proyecta las máscaras binarias de segmentación candidatas.

### 3.2. Función Crítica en el Pipeline de Agarre
En manipulación de ropa sobre mesas de trabajo (ej. mesas metálicas o de madera en laboratorios):
* La tela proyecta sombras complejas sobre la mesa.
* Sin una máscara perfecta de la silueta de la ropa, los algoritmos de detección de bordes seleccionan a menudo la sombra en lugar del tejido real.
* SAM proporciona la máscara binaria $\mathcal{M}_{\text{SAM}} \in \{0, 1\}^{H \times W}$, garantizando que el punto de contacto seleccionado pertenezca al $100\%$ a la prenda y no a la mesa de soporte.

---

## 4. Vision-Language Models (GPT-4o / OWL-ViT): Descubrimiento y Contexto

### 4.1. Desambiguación de Prendas Nuevas
Cuando se introduce una prenda desconocida en la mesa:
1. El VLM identifica la categoría funcional (*"Pantalón de deporte azul"*).
2. Recupera la plantilla de keypoints anatómicos asociados (*cintura, entrepierna, tobillos*).
3. Mediante *Set-of-Marks (SoM)* o detección abierta con **OWL-ViT**, propone bounding boxes iniciales para acotar la búsqueda de los modelos densos, eliminando la confusión simétrica entre el lado izquierdo y el derecho.

---

## 5. Tabla Sinóptica del Ecosistema de Modelos en CLASP

| Modelo Fundacional | Tipo de Arquitectura | Rol Específico en CLASP | Entrada | Salida | Estado de Entrenamiento |
|---|---|---|---|---|---|
| **DINOv2** | ViT-B/14 (86M params) | Correspondencia geométrica local y parches | Imagen RGB ($H \times W \times 3$) | Tensor de parches $\mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times 768}$ | Congelado (*Frozen*) |
| **Stable Diffusion** | U-Net 2D (860M params) | Semántica global y contexto anatómico | Imagen RGB latente + timestep $t$ | Feature maps intermedios $\mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times 1280}$ | Congelado (*Frozen*) |
| **SAM** | ViT-H / Decodificador ligero | Aislamiento binario de la prenda del fondo | Imagen RGB + Punto prompt | Máscara booleana $\mathcal{M} \in \{0, 1\}^{H \times W}$ | Congelado (*Frozen*) |
| **OWL-ViT / GPT-4o** | Transformer Multimodal | Clasificación de prenda y caja de búsqueda | Imagen RGB + Consulta de texto | Bounding Box grueso $\mathcal{B}_{\text{coarse}}$ | Congelado (*Frozen*) |
