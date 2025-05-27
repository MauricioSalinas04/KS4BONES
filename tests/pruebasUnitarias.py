import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import generadorTejidos as gt
from generadorTejidos import TipoTejido
import ks
import numpy as np

def test_generador_tejidos():
    """Prueba unitaria para el generador de tejidos"""
    
    # Test 1: Verificar dimensiones
    imagen = gt.generar_imagen_hueso_simulada(width=100, height=150)
    assert imagen.shape == (150, 100), "Dimensiones incorrectas"
    
    # Test 2: Verificar rango de valores
    assert imagen.min() >= 0 and imagen.max() <= 255, "Valores fuera de rango [0,255]"
    
    # Test 3: Verificar diferencias entre tipos
    img_normal = gt.generar_imagen_hueso_simulada(tipo=TipoTejido.NORMAL, seed=1)
    img_osteo = gt.generar_imagen_hueso_simulada(tipo=TipoTejido.OSTEOPOROSIS, seed=1)
    
    # Debería haber diferencia significativa en intensidad promedio
    diff_promedio = abs(img_normal.mean() - img_osteo.mean())
    assert diff_promedio > 20, "Diferencia insuficiente entre tipos de tejido"
    
    print("✓ Todas las pruebas del generador pasaron")

def test_analisis_ks():
    """Prueba unitaria para el análisis Kolmogorov-Smirnov"""
    
    # Test con imágenes idénticas (no debería rechazar H0)
    imagen_base = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    resultado_identica = ks.prueba_ks_imagenes(imagen_base, imagen_base.copy())
    
    assert not resultado_identica['rechazar_H0'], "Error: rechaza H0 con imágenes idénticas"
    assert resultado_identica['estadistico_D'] < 0.01, "Estadístico D muy alto para imágenes idénticas"
    
    # Test con imágenes muy diferentes
    img1 = np.full((50, 50), 50, dtype=np.uint8)  # Imagen uniforme oscura
    img2 = np.full((50, 50), 200, dtype=np.uint8)  # Imagen uniforme clara
    
    resultado_diferente = ks.prueba_ks_imagenes(img1, img2)
    assert resultado_diferente['rechazar_H0'], "Error: no rechaza H0 con imágenes muy diferentes"
    
    print("✓ Todas las pruebas del análisis KS pasaron")


test_analisis_ks()
test_generador_tejidos()