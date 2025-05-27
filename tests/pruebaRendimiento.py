import sys
import os
import time
import psutil
import numpy as np

# Agregar el directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from generadorTejidos import generar_imagen_hueso_simulada
from ks import prueba_ks_imagenes

def test_rendimiento():
    """Evaluación de rendimiento del sistema"""
    
    tamaños = [(64, 64), (128, 128), (256, 256), (512, 512), (2000, 2000)]
    resultados_rendimiento = []
    
    for width, height in tamaños:
        print(f"Probando rendimiento para {width}x{height}...")
        
        # Medir tiempo de generación
        start_time = time.time()
        img1 = generar_imagen_hueso_simulada(width=width, height=height, seed=1)
        img2 = generar_imagen_hueso_simulada(width=width, height=height, seed=2)
        tiempo_generacion = time.time() - start_time
        
        # Medir memoria antes del análisis
        proceso = psutil.Process(os.getpid())
        memoria_inicial = proceso.memory_info().rss / 1024 / 1024  # MB
        
        # Medir tiempo de análisis
        start_time = time.time()
        resultado = prueba_ks_imagenes(img1, img2)
        tiempo_analisis = time.time() - start_time
        
        # Medir memoria después del análisis
        memoria_final = proceso.memory_info().rss / 1024 / 1024  # MB
        uso_memoria = memoria_final - memoria_inicial
        
        resultados_rendimiento.append({
            'dimensiones': f"{width}x{height}",
            'pixeles': width * height,
            'tiempo_generacion': tiempo_generacion,
            'tiempo_analisis': tiempo_analisis,
            'uso_memoria_mb': uso_memoria
        })
    
    # Mostrar resultados
    print("\nRESULTADOS DE RENDIMIENTO:")
    print("-" * 70)
    print(f"{'Dimensiones':<12} {'Píxeles':<8} {'Gen(s)':<8} {'Análisis(s)':<12} {'Memoria(MB)':<12}")
    print("-" * 70)
    
    for r in resultados_rendimiento:
        print(f"{r['dimensiones']:<12} {r['pixeles']:<8} "
              f"{r['tiempo_generacion']:<8.3f} {r['tiempo_analisis']:<12.3f} "
              f"{r['uso_memoria_mb']:<12.1f}")
    
    return resultados_rendimiento

# Ejecutar pruebas de rendimiento
resultados_perf = test_rendimiento()