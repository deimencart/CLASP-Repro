import numpy as np
import cv2 as cv
from typing import Tuple, Any, Dict, List
import torch

def rotate_image(image: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Image rotation around its geometric center
    Parametros:
      image: Matriz de imagen de entrada (H, W, C) o (H, W).
      angle_deg: Angulo de rotacion en grados (positivo = sentido antihorario).
      
    Retorna:
      Tuple con (imagen_rotada, matriz_afin_2x3).
    """

    h, w = image.shape[:2]
    cx: float = w / 2.0
    cy: float = h / 2.0
    m_rot = cv.getRotationMatrix2D(center=(cx, cy), angle = angle_deg, scale= 1.0 )

    rotated = cv.warpAffine(
    src=image,
    M=m_rot,
    dsize=(w, h),
    flags=cv.INTER_LINEAR,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=0
)
    
    return rotated, m_rot

def backtrack_pixel(
    u_rot: float,
    v_rot: float,
    angle_deg: float,
    shape: Tuple[int, int]
) -> Tuple[int, int]:
    """
    Transforma un pixel detectado en la imagen rotada de vuelta al marco original.
    
    Parametros:
      u_rot: Coordenada horizontal (columna) en la imagen rotada.
      v_rot: Coordenada vertical (fila) en la imagen rotada.
      angle_deg: Angulo con el que se roto la imagen virtual.
      shape: Dimensiones de la imagen (H, W).
      
    Retorna:
      Tuple con (u_real, v_real) en coordenadas de la camara real.
    """
    h, w = shape[:2]
    cx: float = w / 2.0
    cy: float = h / 2.0
    
    # 1. Trasladar el punto al origen del centro de la imagen
    u_shifted: float = u_rot - cx
    v_shifted: float = v_rot - cy
    
    # 2. Convertir el angulo a radianes
    rad: float = float(np.deg2rad(angle_deg))
    cos_theta: float = float(np.cos(rad))
    sin_theta: float = float(np.sin(rad))
    
    # 3. Aplicar rotacion inversa SO(2)
    u_real: float = u_shifted * cos_theta - v_shifted * sin_theta + cx
    v_real: float = u_shifted * sin_theta + v_shifted * cos_theta + cy
    
    # 4. Redondear a enteros y acotar dentro de los limites de la imagen
    u_out: int = int(np.clip(np.round(u_real), 0, w - 1))
    v_out: int = int(np.clip(np.round(v_real), 0, h - 1))
    
    return u_out, v_out


def find_best_orientation(
    image: np.ndarray,
    extractor: Any,
    keypoint_vectors: Dict[str, torch.Tensor],
    sweep_size: int = 336,
    step_deg: int = 30
) -> Tuple[float, float]:
    """
    Descubre automaticamente el angulo optimo de la prenda mediante barrido angular en GPU.
    
    Parametros:
      image: Fotograma original de la camara (H, W, 3).
      extractor: Instancia de DinoV2Extractor.
      keypoint_vectors: Diccionario {nombre: tensor_f_vec} del prototipo canonico.
      sweep_size: Resolucion reducida para acelerar la busqueda (336x336 ejecuta en ~0.7s).
      step_deg: Paso angular en grados (30 grados evalua 12 orientaciones).
      
    Retorna:
      Tuple (best_angle_deg, best_score).
    """
    # 1. Redimensionar copia ligera para el barrido veloz
    img_sweep = cv.resize(image, (sweep_size, sweep_size), interpolation=cv.INTER_LINEAR)
    
    best_angle: float = 0.0
    best_score: float = -1.0
    
    # 2. Barrido sobre el circulo unitario SO(2)
    with torch.inference_mode():
        for angle in range(0, 360, step_deg):
            rot_img, _ = rotate_image(img_sweep, float(angle))
            tensor_rot = extractor.preprocess(rot_img)
            dense_feat = extractor.extract_features(tensor_rot, sweep_size, sweep_size)
            
            # Calcular la confianza media sobre todos los keypoints anotados
            scores: List[float] = []
            for f_vec in keypoint_vectors.values():
                heatmap = torch.einsum("c, chw -> hw", f_vec, dense_feat)
                max_val = float(torch.max(heatmap).item())
                scores.append(max_val)
                
            mean_score = float(sum(scores) / max(len(scores), 1))
            
            if mean_score > best_score:
                best_score = mean_score
                best_angle = float(angle)
                
    return best_angle, best_score



if __name__ == "__main__":
    print("=== TEST DE ALINEACION Y BACKTRACKING ===")
    
    # 1. Prueba matematica cerrada (Round-Trip Test)
    h_test, w_test = 480, 640
    test_angle = 37.5  # Angulo arbitrario en grados
    u_orig, v_orig = 412.0, 185.0
    
    # Calcular posicion rotada con la matriz afin
    cx, cy = w_test / 2.0, h_test / 2.0
    m_test = cv.getRotationMatrix2D((cx, cy), test_angle, 1.0)
    pt_homog = np.array([u_orig, v_orig, 1.0])
    pt_rot = m_test @ pt_homog
    
    # Recuperar el punto original mediante backtrack_pixel
    u_rec, v_rec = backtrack_pixel(pt_rot[0], pt_rot[1], test_angle, (h_test, w_test))
    
    error = np.hypot(u_orig - u_rec, v_orig - v_rec)
    print(f"Punto original: ({u_orig}, {v_orig})")
    print(f"Punto rotado:   ({pt_rot[0]:.2f}, {pt_rot[1]:.2f})")
    print(f"Punto recuperado: ({u_rec}, {v_rec})")
    print(f"Error euclidiano: {error:.4f} pixeles")
    
    assert error <= 1.0, f"Error excesivo en backtracking: {error}"
    print("[PASS] Verificacion matematica superada con exito.")