"""
05_deprojector_3d.py - Estimación de Distancia de Keypoints a la Cámara en Tiempo Real.

Justificación de Primeros Principios:
1. Cada punto semántico detectado en la imagen (u, v) representa un rayo óptico
   que parte del centro de la lente y atraviesa el plano focal.
2. La distancia física desde el centro óptico de la cámara (0, 0, 0) hasta el punto
   en la tela P = [X, Y, Z] corresponde a la norma euclidiana del vector posición:
       R = ||P||_2 = sqrt(X^2 + Y^2 + Z^2)
   donde Z es la profundidad perpendicular al sensor y (X, Y) son las coordenadas
   laterales obtenidas por el modelo Pinhole inverso:
       X = (u - c_x) * Z / (f_x + eps)
       Y = (v - c_y) * Z / (f_y + eps)
3. Si el sensor de profundidad RealSense está activo, Z se obtiene directamente de los
   fotogramas infrarrojos (filtrado con mediana 5x5). Si se usa la webcam estándar,
   se utiliza la distancia nominal aproximada de trabajo (Z ≈ 0.60 m).
4. El script muestra única y exclusivamente la distancia de cada punto semántico a la
   cámara, sin cálculos de medidas de la prenda ni líneas entre puntos.
"""

from typing import Dict, Tuple, List, Optional, Any
import json
import os
import time
from importlib import import_module
import cv2 as cv
import numpy as np
import torch

# Extractor DINOv2 y Módulo de Alineación Angular
DinoV2Extractor = import_module("02_dino_extractor").DinoV2Extractor
aligner_module = import_module("04_rotation_aligner")
rotate_image = aligner_module.rotate_image
backtrack_pixel = aligner_module.backtrack_pixel
find_best_orientation = aligner_module.find_best_orientation

# Paleta BGR para los 7 puntos anatómicos de FoldNet
COLORS: Dict[str, Tuple[int, int, int]] = {
    "l_corner": (0, 0, 255),    # Rojo (Cintura Izq)
    "r_corner": (0, 165, 255),  # Naranja (Cintura Der)
    "crotch":   (0, 255, 255),  # Amarillo (Entrepierna)
    "l_leg_o":  (255, 0, 0),    # Azul (Bajo Exterior Izq)
    "l_leg_i":  (255, 255, 0),  # Cyan (Bajo Interior Izq)
    "r_leg_o":  (255, 0, 255),  # Magenta (Bajo Exterior Der)
    "r_leg_i":  (0, 255, 0),    # Verde (Bajo Interior Der)
}

EPSILON: float = 1e-7


class CameraIntrinsics:
    """Estructura liviana para intrínsecos Pinhole."""
    def __init__(self, fx: float, fy: float, ppx: float, ppy: float, width: int = 640, height: int = 480) -> None:
        self.fx: float = fx
        self.fy: float = fy
        self.ppx: float = ppx
        self.ppy: float = ppy
        self.width: int = width
        self.height: int = height


def setup_camera() -> Tuple[Any, Any, CameraIntrinsics, float, str]:
    """
    Inicializa la cámara RealSense D435 o activa fallback a la webcam estándar.
    """
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("No se detectó cámara Intel RealSense física.")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)

        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        rs_intr = color_stream.get_intrinsics()
        intrinsics = CameraIntrinsics(
            fx=rs_intr.fx, fy=rs_intr.fy, ppx=rs_intr.ppx, ppy=rs_intr.ppy,
            width=rs_intr.width, height=rs_intr.height
        )

        print(f"[OK] Intel RealSense D435 conectada (Escala: {depth_scale:.4f} m/unidad).")
        return pipeline, align, intrinsics, depth_scale, "realsense"

    except Exception as e:
        print(f"[INFO] {e}. Usando webcam estándar con distancia nominal aproximada...")
        cap = cv.VideoCapture(0)
        nominal_intr = CameraIntrinsics(fx=615.0, fy=615.0, ppx=320.0, ppy=240.0, width=640, height=480)
        return cap, None, nominal_intr, 0.001, "webcam"


def deproject_to_distance(
    u: int,
    v: int,
    depth_image: Optional[np.ndarray],
    depth_scale: float,
    intrinsics: CameraIntrinsics,
    window: int = 5,
    default_z: float = 0.60,
) -> Tuple[np.ndarray, float]:
    """
    Calcula la posición 3D [X, Y, Z] y la distancia euclidiana total a la cámara (R en metros).
    """
    z_meter = default_z

    if depth_image is not None:
        h, w = depth_image.shape
        w_half = window // 2
        u_min, u_max = max(0, u - w_half), min(w, u + w_half + 1)
        v_min, v_max = max(0, v - w_half), min(h, v + w_half + 1)
        patch = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = patch[patch > 0]

        if len(valid_depths) > 0:
            z_meter = float(np.median(valid_depths)) * depth_scale

    x_meter = (u - intrinsics.ppx) * z_meter / (intrinsics.fx + EPSILON)
    y_meter = (v - intrinsics.ppy) * z_meter / (intrinsics.fy + EPSILON)

    point_3d = np.array([x_meter, y_meter, z_meter], dtype=np.float32)
    dist_to_camera = float(np.linalg.norm(point_3d))
    return point_3d, dist_to_camera


def load_prototype_vectors(extractor: DinoV2Extractor, proto_img_path: str, proto_json_path: str) -> Dict[str, torch.Tensor]:
    """Carga la plantilla canónica y extrae descriptores DINOv2 unitarios en GPU."""
    proto_img = cv.imread(proto_img_path)
    if proto_img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen prototipo en: {proto_img_path}")

    with open(proto_json_path, "r", encoding="utf-8") as f:
        annotations: Dict[str, List[int]] = json.load(f)

    h_p, w_p = proto_img.shape[:2]
    proto_tensor = extractor.preprocess(proto_img)
    dense_proto = extractor.extract_features(proto_tensor, h_p, w_p)

    keypoint_vectors: Dict[str, torch.Tensor] = {}
    for name, (px, py) in annotations.items():
        keypoint_vectors[name] = dense_proto[:, py, px]

    print(f"[CLASP] Plantilla cargada: {len(keypoint_vectors)} keypoints listos en GPU.")
    return keypoint_vectors


def main() -> None:
    extractor = DinoV2Extractor()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proto_img_path = os.path.join(base_dir, "prototypes", "real_jeans_1.jpg")
    proto_json_path = os.path.join(base_dir, "prototypes", "real_jeans_1.json")

    keypoint_vectors = load_prototype_vectors(extractor, proto_img_path, proto_json_path)
    cam, align, intrinsics, depth_scale, cam_type = setup_camera()
    prev_time = time.time()

    print("\n" + "=" * 65)
    print("ESTIMACIÓN EN TIEMPO REAL: DISTANCIA DE PUNTOS A LA CÁMARA")
    print("Auto-calibración angular activada (0.7s sweep DINOv2)")
    print("Teclas: 'r' recalibrar ángulo | 'q' salir | 's' capturar")
    print("=" * 65 + "\n")

    current_angle: float = 0.0
    calibrated: bool = False
    calib_score: float = 0.0
    frames_without_detection: int = 0

    while True:
        depth_image: Optional[np.ndarray] = None
        if cam_type == "realsense":
            frames = cam.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
        else:
            ret, frame = cam.read()
            if not ret:
                break

        # Auto-calibración angular automática al iniciar o si se pierde la prenda
        if not calibrated:
            print("[AUTO-CALIB] Evaluando orientacion de la prenda en la mesa...")
            current_angle, calib_score = find_best_orientation(frame, extractor, keypoint_vectors)
            calibrated = True
            frames_without_detection = 0
            print(f"[AUTO-CALIB] Orientacion fijada: {current_angle:.1f}° (Similitud: {calib_score:.3f})")

        # 1. Rotar fotograma de inferencia para alinear con el prototipo vertical
        if abs(current_angle) > 1e-3:
            frame_proc, _ = rotate_image(frame, current_angle)
        else:
            frame_proc = frame

        h_q, w_q = frame_proc.shape[:2]

        # 2. Inferencia DINOv2 en GPU sobre el fotograma virtual alineado
        tensor_q = extractor.preprocess(frame_proc)
        dense_query = extractor.extract_features(tensor_q, h_q, w_q)

        # Distancias detectadas a la cámara
        points_telemetry: List[Tuple[str, float, Tuple[int, int]]] = []

        for name, f_vec in keypoint_vectors.items():
            heatmap = torch.einsum("c, chw -> hw", f_vec, dense_query)
            max_idx = torch.argmax(heatmap).item()
            best_y_rot = int(max_idx // w_q)
            best_x_rot = int(max_idx % w_q)
            score = float(heatmap[best_y_rot, best_x_rot].item())

            if score >= 0.60:
                # 3. BACKTRACKING: Devolver el píxel al marco real no rotado de la cámara
                if abs(current_angle) > 1e-3:
                    u_real, v_real = backtrack_pixel(best_x_rot, best_y_rot, current_angle, frame.shape[:2])
                else:
                    u_real, v_real = best_x_rot, best_y_rot

                # 4. Desproyección 3D usando el píxel real y el mapa de profundidad original
                _, dist_camera = deproject_to_distance(
                    u=u_real, v=v_real,
                    depth_image=depth_image,
                    depth_scale=depth_scale,
                    intrinsics=intrinsics,
                    window=5,
                    default_z=0.60,
                )
                points_telemetry.append((name, dist_camera, (u_real, v_real)))

                # Dibujar punto en la imagen original
                color = COLORS.get(name, (0, 255, 0))
                cv.circle(frame, (u_real, v_real), 6, color, -1)
                cv.circle(frame, (u_real, v_real), 9, (255, 255, 255), 1)

                # Etiqueta limpia: SOLO nombre y distancia a la cámara
                label = f"{name}: {dist_camera:.2f} m"
                cv.putText(frame, label, (u_real + 10, v_real - 6),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

        # Disparo de recalibración si la prenda se ha movido o girado (30 frames sin detecciones)
        if len(points_telemetry) == 0:
            frames_without_detection += 1
            if frames_without_detection >= 45:
                print("[AUTO-CALIB] Puntos no detectados; reescaneando orientacion...")
                calibrated = False
        else:
            frames_without_detection = 0

        # Panel HUD Lateral: Lista de distancias a la cámara y orientación activa
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + EPSILON)
        prev_time = curr_time

        hud_h = 50 + max(1, len(points_telemetry)) * 22
        cv.rectangle(frame, (10, 10), (320, hud_h), (20, 20, 20), -1)
        cv.rectangle(frame, (10, 10), (320, hud_h), (100, 100, 100), 1)

        cv.putText(frame, f"Distancia | {fps:.1f} FPS | Ang: {current_angle:.0f}°", (18, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv.LINE_AA)

        for idx, (name, dist_m, _) in enumerate(points_telemetry):
            cv.putText(frame, f"{name:10s}: {dist_m:.2f} m ({dist_m*100:.0f} cm)",
                       (18, 52 + idx * 22),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)

        if not points_telemetry:
            cv.putText(frame, "Buscando keypoints...", (18, 52),
                       cv.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv.LINE_AA)

        cv.imshow("CLASP - Distancia de Puntos a la Camara", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] Recalibracion manual solicitada...")
            calibrated = False
        elif key == ord('s'):
            out_snap = os.path.join(base_dir, "output", f"snap_dist_{int(time.time())}.png")
            os.makedirs(os.path.dirname(out_snap), exist_ok=True)
            cv.imwrite(out_snap, frame)
            print(f"[SNAPSHOT] Guardado: {out_snap}")

    if cam_type == "realsense":
        cam.stop()
    else:
        cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
