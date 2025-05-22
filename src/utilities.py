import numpy as np
import cv2
import json
from typing import Dict

# Funciones de utilidad para cargar y usar el detector
def load_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen desde archivo
    
    Args:
        image_path: Ruta del archivo de imagen
        
    Returns:
        Imagen en escala de grises como array numpy
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    return image

def save_report(report: Dict, output_path: str):
    """
    Guarda el reporte en formato JSON
    
    Args:
        report: Diccionario con el reporte
        output_path: Ruta donde guardar el archivo
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

