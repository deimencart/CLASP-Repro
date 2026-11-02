"""
clasp_interactive_perception.py - Arquitectura de Percepción Interactiva (IP) para CLASP.

========================================================================================
JUSTIFICACIÓN DE PRIMEROS PRINCIPIOS Y FUENTES MATEMÁTICAS
========================================================================================

1. FUENTE 1: CLASP (Clothes Manipulation with Semantic Keypoints - Deng & Hsu, 2025/2026)
   - Postulado: Los keypoints semánticos anatómicos (l_corner, r_corner, crotch, etc.)
     constituyen los "affordances" funcionales primarios de manipulación.
   - Limitación resuelta: CLASP asume ejecución en bucle abierto (Open-Loop). Si la tela se
     mueve o se arruga, la pinza pellizca el vacío.

2. FUENTE 2: CeDiRNet-3DoF (Center Direction Network for Grasping - ICRA 2024)
   - Postulado: Un agarre textil requiere posición (X, Y, Z) Y ángulo de ataque (theta_yaw).
     Alinear las mordazas a 90 grados respecto a la costura previene el deslizamiento.

3. FUENTE 3: Interactive Perception & Active Perception (MIT Robotic Manipulation - Tedrake 2024)
   - Postulado: Reducción activa de entropía: H(V_true | S_obs). Ante oclusiones o baja confianza,
     el robot sondea físicamente o ajusta el ángulo de visión antes del compromiso del pliegue.

4. FUENTE 4: Ramisa et al. (Determining Where to Grasp Cloth - CCIA 2011)
   - Postulado: Monitoreo de normales locales y detección de contacto en superficie para
     garantizar la inserción de las mordazas entre la tela y la mesa.

========================================================================================
FORMALISMO MATEMÁTICO INTEGRADO
========================================================================================

A. Transformación Rígida SE(3) Cámara -> Base del Robot:
   [P_base; 1] = T_base_cam * [P_cam; 1]
   T_base_cam = [ R_cam_base,  t_cam_base ]
                [    0,             1     ]  # shape: (4, 4)

B. Control Proporcional de Aproximación Aérea (Hover):
   e(t) = (P_target_base + h_hover * z_hat) - x_TCP(t)
   dot(x)_cmd = K_p * e(t)

C. Criterio de Parada de Touchdown Háptico (Guard Condition):
   Detener descenso cuando:
   F_z(t) >= F_trigger  O BIEN  z_TCP(t) <= z_table + delta_safety

D. Discriminación Propioceptiva de Agarre (Apertura de Mordazas Robotiq 2F-85):
   theta_jaw = ángulo articular del motor (0.0 rad = abierto 85mm, 0.85 rad = cerrado 0mm)
   - Si theta_jaw >= 0.80 rad (cierre > 95%): FALLO (Pinza vacía)
   - Si 0.60 <= theta_jaw <= 0.75 rad: ÉXITO (Tela atrapada)
   - Si theta_jaw < 0.60 rad: BLOQUEO MECÁNICO (Colisión prematura)

E. Invariante Cinemático de Verificación Visual (Lift & Verify):
   || delta_P_keypoint - delta_x_TCP ||_2 <= epsilon_slip
"""

from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import time

EPSILON: float = 1e-7


@dataclass
class ClaspPerceptionPacket:
    """Paquete de telemetría emitido en tiempo real por el pipeline CLASP."""
    target_name: str
    pos_camera: np.ndarray        # shape: (3,), coordenadas métricas [X, Y, Z] respecto a la lente
    confidence: float             # score de similitud coseno de DINOv2 en [0, 1]
    timestamp: float              # tiempo epoch para monitor de vigencia temporal (watchdog)
    normal_vector: np.ndarray     # shape: (3,), vector normal al contorno o [0, 0, 1]


class ExtrinsicCalibrator:
    """Transformador de coordenadas homogéneas SE(3) entre la cámara y la base del robot."""

    def __init__(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> None:
        """
        Args:
            rotation_matrix: Matriz SO(3) de rotación cámara -> base. # shape: (3, 3)
            translation_vector: Vector de traslación cámara -> base en metros. # shape: (3,)
        """
        assert rotation_matrix.shape == (3, 3), "La matriz de rotación debe ser 3x3"
        assert translation_vector.shape == (3,), "El vector de traslación debe ser de dimensión 3"

        self.T_base_cam = np.eye(4, dtype=np.float32)  # shape: (4, 4)
        self.T_base_cam[:3, :3] = rotation_matrix
        self.T_base_cam[:3, 3] = translation_vector

    def transform_camera_to_base(self, p_cam: np.ndarray) -> np.ndarray:
        """
        Proyecta un punto cartesiano de la cámara a coordenadas métricas de la base del robot.
        
        Args:
            p_cam: Coordenadas métricas [X, Y, Z] en marco cámara. # shape: (3,)
            
        Returns:
            p_base: Coordenadas métricas [X, Y, Z] en marco base del robot. # shape: (3,)
        """
        assert p_cam.shape == (3,), "El punto de entrada debe tener dimensión 3"
        homo_cam = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float32)  # shape: (4,)
        homo_base = self.T_base_cam @ homo_cam                                       # shape: (4,)
        return homo_base[:3]


class InteractivePerceptionController:
    """
    Controlador de Percepción Interactiva (IP):
    Gestiona la aproximación en bucle cerrado, el touchdown háptico y la verificación de agarre.
    """

    def __init__(
        self,
        calibrator: ExtrinsicCalibrator,
        hover_height: float = 0.060,     # 6 cm sobre el punto
        touchdown_offset: float = 0.010,  # 1 cm sobre la superficie de la mesa
        table_z_world: float = 0.430,     # Cota Z de Mesa 3 según scene_attributes.py
        force_threshold_n: float = 3.0,   # Disparo de parada por fuerza (3 N)
    ) -> None:
        self.calibrator = calibrator
        self.hover_height = hover_height
        self.touchdown_offset = touchdown_offset
        self.table_z_world = table_z_world
        self.force_threshold_n = force_threshold_n

    def compute_approach_hover(self, p_cam: np.ndarray, yaw_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la pose de aproximación aérea (Pre-Grasp Hover) en la base del robot
        con orientación de mordaza alineada normal al borde de la tela (yaw_rad).
        
        Args:
            p_cam: Coordenadas métricas [X, Y, Z] respecto a la cámara. # shape: (3,)
            yaw_rad: Ángulo de aproximación antipodal sobre el plano horizontal (rad).
            
        Returns:
            p_hover: Vector cartesiano [X, Y, Z] a +hover_height sobre el objetivo. # shape: (3,)
            q_hover: Cuaternión de orientación [w, x, y, z] con TCP hacia abajo y rotación yaw. # shape: (4,)
        """
        p_base = self.calibrator.transform_camera_to_base(p_cam)
        p_hover = p_base.copy()
        p_hover[2] += self.hover_height

        # Cuaternión analítico R_z(yaw) * R_y(pi):
        # Eje local Z apunta verticalmente hacia abajo [0, 0, -1]
        # Eje local X gira según yaw_rad para entrar perpendicular al filo
        half_yaw = yaw_rad / 2.0
        q_hover = np.array([
            0.0,
            -np.sin(half_yaw),
            np.cos(half_yaw),
            0.0
        ], dtype=np.float32)
        # Normalización de seguridad con epsilon
        q_norm = np.linalg.norm(q_hover) + EPSILON
        q_hover = q_hover / q_norm

        return p_hover, q_hover

    def compute_approach_hover_from_target(self, target: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula la pose aérea directamente a partir de una instancia de HybridGraspTarget.
        
        Args:
            target: Instancia con atributos 'point_cam' y 'yaw_rad'.
            
        Returns:
            p_hover: Vector cartesiano [X, Y, Z] a +hover_height. # shape: (3,)
            q_hover: Cuaternión de aproximación [w, x, y, z]. # shape: (4,)
        """
        yaw = getattr(target, "yaw_rad", 0.0)
        p_cam = getattr(target, "point_cam")
        return self.compute_approach_hover(p_cam, yaw_rad=yaw)

    def evaluate_touchdown_guard(self, current_z: float, force_z: float) -> Tuple[bool, str]:
        """
        Condición de parada háptica/cinemática para el descenso hacia la tela.
        
        Args:
            current_z: Altura Z actual del TCP del robot en metros.
            force_z: Fuerza normal medida por el sensor F/T en Newtons.
            
        Returns:
            should_stop: True si se debe detener el descenso.
            reason: Causa de la detención ('FORCE_TRIGGER', 'TABLE_LIMIT', 'DESCENDING').
        """
        if force_z >= self.force_threshold_n:
            return True, "FORCE_TRIGGER"

        z_limit = self.table_z_world + self.touchdown_offset
        if current_z <= z_limit:
            return True, "TABLE_LIMIT"

        return False, "DESCENDING"

    def verify_grasp_proprioception(self, jaw_angle_rad: float) -> Tuple[bool, str]:
        """
        Verifica si la pinza atrapó tejido examinando la apertura final de las mordazas.
        
        Args:
            jaw_angle_rad: Ángulo articular de la Robotiq 2F-85 (0.0 rad = abierto, 0.85 rad = cerrado).
            
        Returns:
            is_grasped: True si hay tela atrapada con firmeza.
            status: Diagnóstico ('SUCCESS', 'EMPTY_AIR', 'JAMMED').
        """
        if jaw_angle_rad >= 0.80:
            return False, "EMPTY_AIR"  # Pinza se cerró casi al 100%, no atrapó tela

        if 0.60 <= jaw_angle_rad < 0.80:
            return True, "SUCCESS"     # Pinza se detuvo en el espesor típico del tejido (0.70 rad)

        return False, "JAMMED"         # Pinza bloqueada a más de la mitad, choque prematuro

    def verify_lift_kinematics(
        self,
        tcp_displacement: np.ndarray,
        keypoint_displacement: np.ndarray,
        tolerance_m: float = 0.015,
    ) -> bool:
        """
        Lift & Verify: Comprueba si el keypoint se movió solidariamente con la pinza.
        """
        error = float(np.linalg.norm(tcp_displacement - keypoint_displacement))
        return error <= tolerance_m


# ==============================================================================
# MINI-TEST UNITARIO DE VERIFICACIÓN FEYNMAN
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("EJECUTANDO MINI-TEST DE ARQUITECTURA CLASP + INTERACTIVE PERCEPTION")
    print("=" * 70)

    # 1. Calibrador Extrínseco Sintético (Cámara sobre Mesa 3)
    R_dummy = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float32)
    t_dummy = np.array([0.15, -0.70, 1.20], dtype=np.float32)  # Cámara a 1.20m de altura
    calib = ExtrinsicCalibrator(R_dummy, t_dummy)

    # 2. Instanciar Controlador IP
    controller = InteractivePerceptionController(
        calibrator=calib,
        hover_height=0.060,
        table_z_world=0.430,
        force_threshold_n=3.0,
    )

    # Test A: Transformación Pinhole a Base y Hover con Rotación Yaw
    p_cam_mock = np.array([0.05, -0.10, 0.65], dtype=np.float32)  # Keypoint detectado por DINOv2
    yaw_mock = np.radians(45.0)                                    # Ataque oblicuo a 45 grados
    p_hover, q_hover = controller.compute_approach_hover(p_cam_mock, yaw_rad=yaw_mock)
    print(f"[TEST A] P_cam: {p_cam_mock} -> P_hover en Base: {p_hover}")
    print(f"         Cuaternión Yaw 45°: {q_hover}")
    assert np.isfinite(p_hover).all(), "P_hover contiene NaNs o infinitos"
    assert p_hover[2] > 0.430, "P_hover debe estar por encima de la mesa"
    assert abs(np.linalg.norm(q_hover) - 1.0) < 1e-5, "El cuaternión de hover debe ser unitario"

    # Test B: Sonda de Contacto Háptica
    stop_free, r_free = controller.evaluate_touchdown_guard(current_z=0.550, force_z=0.5)
    assert not stop_free and r_free == "DESCENDING", "Descenso libre falló"

    stop_force, r_force = controller.evaluate_touchdown_guard(current_z=0.450, force_z=3.8)
    assert stop_force and r_force == "FORCE_TRIGGER", "Disparo por fuerza falló"

    stop_table, r_table = controller.evaluate_touchdown_guard(current_z=0.438, force_z=1.0)
    assert stop_table and r_table == "TABLE_LIMIT", "Tope de seguridad por cota de mesa falló"
    print(f"[TEST B] Touchdown Háptico validado con éxito.")

    # Test C: Verificación Propioceptiva de Agarre
    grasped_ok, stat_ok = controller.verify_grasp_proprioception(jaw_angle_rad=0.70)
    assert grasped_ok and stat_ok == "SUCCESS", "Detección de tela falló"

    grasped_empty, stat_empty = controller.verify_grasp_proprioception(jaw_angle_rad=0.83)
    assert not grasped_empty and stat_empty == "EMPTY_AIR", "Detección de pinza vacía falló"
    print(f"[TEST C] Discriminación de agarre por apertura validada con éxito.")

    # Test D: Lift & Verify
    d_tcp = np.array([0.0, 0.0, 0.030], dtype=np.float32)  # Pinza sube 3 cm
    d_kp_ok = np.array([0.002, 0.001, 0.028], dtype=np.float32)  # Tela sube solidaria
    assert controller.verify_lift_kinematics(d_tcp, d_kp_ok), "Lift & Verify falló en caso afirmativo"

    d_kp_slip = np.array([0.0, 0.0, 0.000], dtype=np.float32)  # Tela se quedó en la mesa
    assert not controller.verify_lift_kinematics(d_tcp, d_kp_slip), "Lift & Verify falló al detectar deslizamiento"
    print(f"[TEST D] Invariante cinemático de Lift & Verify superado.")

    print("\nTODOS LOS TESTS DE ARQUITECTURA CLASP + IP SUPERADOS CON ÉXITO [PASS].")
