import numpy as np
import cv2
import json
from typing import Dict
from datetime import datetime
import os

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

def generate_preprocessing_report(metrics: Dict, output_dir: str) -> None:
    """
    Genera informe markdown de verificación de preprocesamiento
    
    Args:
        metrics: Diccionario con métricas de calidad
        output_dir: Directorio donde guardar el informe
    """
    report_path = os.path.join(output_dir, 'preprocessing_report.md')
    total_checks = 0
    passed_checks = 0
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Encabezado
        f.write('# Informe de Verificación de Preprocesamiento\n\n')
        f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        # Resumen de calidad
        f.write('## Resumen de Calidad\n\n')
        
        # Iterar por cada fase
        for phase, values in metrics.items():
            if phase != 'quality_thresholds':
                f.write(f'## {phase.title()}\n\n')
                f.write('| Métrica | Valor | Rango Esperado | Estado |\n')
                f.write('|---------|--------|----------------|--------|\n')
                
                # Modificar la sección de verificación de threshold:
                thresholds = metrics['quality_thresholds'][phase]
                for metric, value in values.items():
                    threshold = thresholds.get(metric, None)
                    if threshold is not None:  # Cambio aquí
                        total_checks += 1
                        if isinstance(threshold, tuple):
                            status = '✅' if threshold[0] <= value <= threshold[1] else '❌'
                            range_str = f'{threshold[0]:.2f} - {threshold[1]:.2f}'
                            if threshold[0] <= value <= threshold[1]:
                                passed_checks += 1
                        else:
                            status = '✅' if value == threshold else '❌'
                            range_str = f'{threshold}'
                            if value == threshold:
                                passed_checks += 1
                    else:  # threshold is None
                        status = '➖'
                        range_str = 'No definido'
                    
                    f.write(f'| {metric} | {value:.3f} | {range_str} | {status} |\n')
                f.write('\n')
        
        # Score de calidad global
        quality_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        f.write(f'\n## Score de Calidad Global\n\n')
        f.write(f'- Checks pasados: {passed_checks}/{total_checks}\n')
        f.write(f'- Score de calidad: {quality_score:.1f}%\n\n')
        
        # Lista de imágenes generadas
        f.write('## Imágenes Generadas\n\n')
        f.write('1. `1_normalized.png`: Imagen normalizada\n')
        f.write('2. `2_equalized.png`: Imagen ecualizada\n')
        f.write('3. `3_enhanced.png`: Realce trabecular\n')
        f.write('4. `4_bone_mask.png`: Máscara ósea\n')
        f.write('5. `5_final.png`: Imagen final procesada\n')

    # Agregar score de calidad a las métricas
    metrics['quality_score'] = {
        'total_checks': total_checks,
        'passed_checks': passed_checks,
        'score_percentage': quality_score
    }


