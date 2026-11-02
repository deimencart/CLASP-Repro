import torch 
import torch.nn.functional as F
import torchvision.transforms as T
import cv2 as cv
import numpy as np
from typing import Optional, Union

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DinoV2Extractor: 
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        self.device = torch.device(device if device is not None else DEFAULT_DEVICE)
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor: 
        """
        Preprocess de la image para pasarla por DinoV2
        """
        H_orig, W_orig , C_orig = img_bgr.shape
        H_new = max((H_orig // 14) * 14, 14)
        W_new = max((W_orig // 14) * 14, 14)
        img_resize = cv.resize(img_bgr, (W_new, H_new))
        img_new_c = cv.cvtColor(img_resize, cv.COLOR_BGR2RGB)
        #converting numpy matrix to tensor of Pytorch and normalize 
        img_tensor = T.ToTensor()(img_new_c)
        normalize = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
        )
        img_tensor = normalize(img_tensor)
        # 1. Añade una nueva dimensión de tamaño 1 en la posición 0:
        # Pasa de tener forma (3, H, W) a tener forma (1, 3, H, W)
        batch_tensor = img_tensor.unsqueeze(0)

        # 2. Transfiere el tensor a la memoria del dispositivo (VRAM si es CUDA o RAM si es CPU)
        batch_tensor = batch_tensor.to(self.device)

        # 3. Lo devuelves
        return batch_tensor

    
    @torch.inference_mode()
    def extract_features(
        self, tensor: torch.Tensor, orig_h: int, orig_w: int
    ) -> torch.Tensor:
        """Extrae mapa denso de descriptores DINOv2 interpolados a resolución original.

        Args:
            tensor: Imagen preprocesada de forma (1, 3, H_new, W_new).
            orig_h: Alto de la imagen original en píxeles.
            orig_w: Ancho de la imagen original en píxeles.

        Returns:
            Tensor normalizado de forma (384, orig_h, orig_w) con norma L2 unitaria.
        """
        # 1. Obtener dimensiones de la cuadrícula de parches (14x14)
        _, _, h_new, w_new = tensor.shape
        h_patches = h_new // 14
        w_patches = w_new // 14

        # 2. Inferencia en el modelo para obtener tokens de parches
        # shape salida: (1, N_parches, 384) donde N_parches = h_patches * w_patches
        features_dict = self.model.forward_features(tensor)
        patch_tokens: torch.Tensor = features_dict["x_norm_patchtokens"]

        # 3. Reconstruir cuadrícula espacial 2D
        # Paso 3a: agrupar parches en filas y columnas -> shape: (1, h_patches, w_patches, 384)
        features_2d = patch_tokens.reshape(1, h_patches, w_patches, 384)
        # Paso 3b: transponer canales al eje 1 (B, C, H, W) -> shape: (1, 384, h_patches, w_patches)
        features_2d = features_2d.permute(0, 3, 1, 2)

        # 4. Interpolar a la resolución de la imagen original
        # shape salida: (1, 384, orig_h, orig_w)
        features_interp = F.interpolate(
            features_2d,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        # 5. Normalización L2 a lo largo de los 384 canales (dim=1) con protección epsilon
        # shape salida: (1, 384, orig_h, orig_w)
        features_norm = F.normalize(features_interp, p=2, dim=1, eps=1e-7)

        # 6. Retirar dimensión de batch (dim 0) -> shape final: (384, orig_h, orig_w)
        return features_norm.squeeze(0)

    

if __name__ == "__main__":
    extractor = DinoV2Extractor()

    # Imagen simulada con OpenCV (H=480, W=640, C=3)
    dummy_img_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # 1. Preprocesar
    orig_h, orig_w = dummy_img_bgr.shape[:2]
    input_tensor = extractor.preprocess(dummy_img_bgr)

    # 2. Extraer características
    dense_features = extractor.extract_features(input_tensor, orig_h, orig_w)

    # Aserciones físicas y matemáticas
    assert dense_features.shape == (384, 480, 640), f"Shape errónea: {dense_features.shape}"
    assert not torch.isnan(dense_features).any(), "Existen NaNs en los descriptores"

    # Verificar que la norma L2 en cualquier píxel es exactamente 1.0
    pixel_norm = torch.norm(dense_features[:, 100, 100], p=2)
    assert torch.isclose(pixel_norm, torch.tensor(1.0, device=dense_features.device), atol=1e-5), \
        f"Norma no unitaria: {pixel_norm.item()}"

    print(f"Éxito: Mapa de descriptores generado correctamente con forma {dense_features.shape}")