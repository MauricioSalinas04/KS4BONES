import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def visualize_results(normal_image: np.ndarray, test_image: np.ndarray, 
                     results: Dict, save_path: str = None):
    """
    Visualiza los resultados del análisis
    
    Args:
        normal_image: Imagen normal
        test_image: Imagen de prueba
        results: Resultados del análisis
        save_path: Ruta para guardar la visualización (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Imagen normal
    axes[0, 0].imshow(normal_image, cmap='gray')
    axes[0, 0].set_title('Imagen Normal')
    axes[0, 0].axis('off')
    
    # Imagen de prueba
    axes[0, 1].imshow(test_image, cmap='gray')
    axes[0, 1].set_title('Imagen de Prueba')
    axes[0, 1].axis('off')
    
    # Histogramas
    axes[1, 0].hist(normal_image.flatten(), bins=50, alpha=0.7, label='Normal', density=True)
    axes[1, 0].hist(test_image.flatten(), bins=50, alpha=0.7, label='Prueba', density=True)
    axes[1, 0].set_title('Histogramas de Intensidad')
    axes[1, 0].legend()
    
    # Información del análisis
    composite_score = results.get('composite_score', {})
    score_text = f"""
    Score Compuesto: {composite_score.get('score', 0):.3f}
    Confianza: {composite_score.get('confidence', 0):.3f}
    
    Interpretación:
    {composite_score.get('interpretation', 'N/A')}
    
    KS Global: {results.get('global_ks', {}).get('distance', 0):.3f}
    Regiones Significativas: {results.get('regional_ks', {}).get('regions_significant', 0)}
    """
    
    axes[1, 1].text(0.1, 0.9, score_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, fontfamily='monospace')
    axes[1, 1].set_title('Resultados del Análisis')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()