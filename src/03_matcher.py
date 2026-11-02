"""
03_matcher.py - Matching por Similitud Coseno de Keypoints Semánticos (CLASP).

Compara los vectores unitarios L2 extraídos de la prenda prototipo contra el mapa
denso de descriptores de una imagen de consulta para localizar las partes anatómicas
de la prenda mediante producto punto vectorizado.
"""

import json
import os
import importlib
from typing import Dict, Any, Tuple, List, Optional
import cv2 as cv
import numpy as np
import torch

# Importación dinámica del extractor (debido al prefijo numérico en el nombre del archivo)
dino_module = importlib.import_module("02_dino_extractor")
DinoV2Extractor = dino_module.DinoV2Extractor


class SemanticMatcher:
    """Calculador de correspondencia semántica densa basado en DINOv2 y similitud coseno."""

    def __init__(self, extractor: Optional[DinoV2Extractor] = None) -> None:
        """Inicializa el extractor de características en GPU o CPU."""
        self.extractor = extractor if extractor is not None else DinoV2Extractor()

    def match_keypoint(
        self,
        f_keypoint: torch.Tensor,
        dense_query_features: torch.Tensor,
    ) -> Tuple[int, int, float, torch.Tensor]:
        """Calcula el mapa de calor de similitud coseno y localiza el píxel óptimo (x*, y*).

        Fundamento Físico / Matemático:
        Al tener f_keypoint y dense_query_features norma L2 unitaria (||f||_2 = 1.0),
        la similitud coseno en cada píxel (y, x) se reduce al producto escalar estricto:
            S(y, x) = f_keypoint · dense_query_features[:, y, x]
        El máximo de este campo escalar indica la correspondencia anatómica más verosímil.

        Args:
            f_keypoint: Vector descriptor unitario del punto prototipo, shape: (384,).
            dense_query_features: Mapa de descriptores de consulta, shape: (384, H, W).

        Returns:
            best_x: Columna del píxel óptimo en la imagen de consulta.
            best_y: Fila del píxel óptimo en la imagen de consulta.
            max_score: Puntuación de similitud coseno en [-1.0, 1.0].
            heatmap: Tensor 2D con el mapa de activación (H, W).
        """
        # Producto punto denso vectorizado en GPU sin bucles manuales (shape: (H, W))
        heatmap = torch.einsum("c, chw -> hw", f_keypoint, dense_query_features)

        h_query, w_query = heatmap.shape
        max_idx = torch.argmax(heatmap).item()

        # Desenrollar índice plano a coordenadas 2D (y, x)
        best_y = int(max_idx // w_query)
        best_x = int(max_idx % w_query)
        max_score = float(heatmap[best_y, best_x].item())

        return best_x, best_y, max_score, heatmap

    def match_all(
        self,
        prototype_bgr: np.ndarray,
        prototype_annotations: Dict[str, List[int]],
        query_bgr: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Extrae descriptores de ambas imágenes y localiza todos los keypoints anotados.

        Args:
            prototype_bgr: Imagen BGR de la prenda canónica (H_p, W_p, 3).
            prototype_annotations: Diccionario {nombre_keypoint: [x, y]}.
            query_bgr: Imagen BGR de la prenda de consulta (H_q, W_q, 3).

        Returns:
            Diccionario de predicciones con coordenadas [x, y], score y heatmap por keypoint.
        """
        h_proto, w_proto = prototype_bgr.shape[:2]
        h_query, w_query = query_bgr.shape[:2]

        # 1. Extracción de mapas densos de descriptores (shape: (384, H, W))
        proto_tensor = self.extractor.preprocess(prototype_bgr)
        dense_proto = self.extractor.extract_features(proto_tensor, h_proto, w_proto)

        query_tensor = self.extractor.preprocess(query_bgr)
        dense_query = self.extractor.extract_features(query_tensor, h_query, w_query)

        predictions: Dict[str, Dict[str, Any]] = {}

        # 2. Bucle sobre cada punto semántico anotado
        for keypoint_name, coord in prototype_annotations.items():
            px, py = int(coord[0]), int(coord[1])

            # Protección de límites en el prototipo
            px = max(0, min(px, w_proto - 1))
            py = max(0, min(py, h_proto - 1))

            # Extraer vector descriptor puntual (384,)
            f_keypoint = dense_proto[:, py, px]

            # Matching contra la imagen de consulta
            best_x, best_y, score, heatmap = self.match_keypoint(f_keypoint, dense_query)

            predictions[keypoint_name] = {
                "proto_coord": [px, py],
                "query_coord": [best_x, best_y],
                "score": score,
                "heatmap": heatmap.detach().cpu().numpy(),
            }

        return predictions

    def visualize_and_save(
        self,
        prototype_bgr: np.ndarray,
        query_bgr: np.ndarray,
        predictions: Dict[str, Dict[str, Any]],
        output_path: str,
    ) -> np.ndarray:
        """Genera un lienzo comparativo lado a lado con las detecciones visuales."""
        # Altura común para el lienzo lado a lado
        h_p, w_p = prototype_bgr.shape[:2]
        h_q, w_q = query_bgr.shape[:2]
        target_h = max(h_p, h_q)

        # Escalar ambas imágenes a la misma altura manteniendo relación de aspecto
        scale_p = target_h / h_p
        scale_q = target_h / h_q

        p_resized = cv.resize(prototype_bgr, (int(w_p * scale_p), target_h))
        q_resized = cv.resize(query_bgr, (int(w_q * scale_q), target_h))

        canvas = np.hstack([p_resized, q_resized])
        offset_x = p_resized.shape[1]

        # Paleta de colores consistente por keypoint
        palette = [
            (0, 255, 0),    # Verde (l_corner)
            (0, 165, 255),  # Naranja (r_corner)
            (255, 0, 0),    # Azul (crotch)
            (255, 0, 255),  # Magenta (l_leg_o)
            (255, 255, 0),  # Cyan (l_leg_i)
            (0, 255, 255),  # Amarillo (r_leg_i)
            (0, 0, 255),    # Rojo (r_leg_o)
        ]

        for i, (name, data) in enumerate(predictions.items()):
            color = palette[i % len(palette)]

            # Coordenadas en prototipo
            px = int(data["proto_coord"][0] * scale_p)
            py = int(data["proto_coord"][1] * scale_p)

            # Coordenadas en consulta
            qx = int(data["query_coord"][0] * scale_q) + offset_x
            qy = int(data["query_coord"][1] * scale_q)
            score = data["score"]

            # Dibujar puntos y etiquetas
            cv.circle(canvas, (px, py), 6, color, -1)
            cv.putText(canvas, name, (px + 8, py - 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv.putText(canvas, name, (px + 8, py - 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv.circle(canvas, (qx, qy), 6, color, -1)
            label_q = f"{name} ({score:.2f})"
            cv.putText(canvas, label_q, (qx + 8, qy - 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv.putText(canvas, label_q, (qx + 8, qy - 4), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Línea conectora entre prototipo y predicción
            cv.line(canvas, (px, py), (qx, qy), color, 1, cv.LINE_AA)

        # Encabezados de columnas
        cv.putText(canvas, "[PROTOTIPO] real_jeans_1", (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(canvas, "[CONSULTA] Prediccion CLASP", (offset_x + 20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv.imwrite(output_path, canvas)
        return canvas


# ==============================================================================
# PRUEBA Y VERIFICACIÓN
# ==============================================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proto_img_path = os.path.join(base_dir, "prototypes", "real_jeans_1.jpg")
    proto_json_path = os.path.join(base_dir, "prototypes", "real_jeans_1.json")
    query_img_path = os.path.join(base_dir, "samples", "real_jeans_2.jpg")
    output_img_path = os.path.join(base_dir, "output", "matched_result.png")

    print("[1/4] Cargando imagenes y anotaciones...")
    proto_img = cv.imread(proto_img_path)
    assert proto_img is not None, f"No se pudo cargar {proto_img_path}"

    query_img = cv.imread(query_img_path)
    assert query_img is not None, f"No se pudo cargar {query_img_path}"

    with open(proto_json_path, "r", encoding="utf-8") as f:
        proto_annotations = json.load(f)

    print(f"  Prototipo: {proto_img.shape} | Consulta: {query_img.shape}")
    print(f"  Keypoints canónicos FoldNet a detectar: {list(proto_annotations.keys())}")

    print("[2/4] Ejecutando extractor e inferencia de similitud coseno...")
    matcher = SemanticMatcher()
    results = matcher.match_all(proto_img, proto_annotations, query_img)

    print("\n" + "=" * 65)
    print(f"{'Keypoint':<12} | {'Prototipo (X, Y)':<18} | {'Prediccion (X, Y)':<18} | {'Score':<6}")
    print("=" * 65)

    q_h, q_w = query_img.shape[:2]
    for kp_name, data in results.items():
        px, py = data["proto_coord"]
        qx, qy = data["query_coord"]
        score = data["score"]

        print(f"{kp_name:<12} | [{px:>3}, {py:>3}]            | [{qx:>3}, {qy:>3}]            | {score:.3f}")

        # Aserciones de verificación física y matemática
        assert 0 <= qx < q_w, f"Predicción X={qx} fuera de rango [0, {q_w})"
        assert 0 <= qy < q_h, f"Predicción Y={qy} fuera de rango [0, {q_h})"
        assert -1.0 <= score <= 1.0, f"Score de similitud fuera de rango cosenoidal: {score}"
        assert not np.isnan(score), f"Score NaN detectado en {kp_name}"

    print("=" * 65)

    print(f"\n[3/4] Generando visualizacion comparativa en: {output_img_path}")
    matcher.visualize_and_save(proto_img, query_img, results, output_img_path)

    assert os.path.exists(output_img_path), "Fallo al guardar la imagen de salida"
    print("[4/4] Verificacion completada con EXITO. Cero NaNs y coordenadas en rango.")
