import sys
import os
import timeit

# Agregar directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from generadorTejidos import generar_imagen_hueso_simulada
from ks import prueba_ks_imagenes

def benchmark_completo():
    """Prueba de rendimiento completa del sistema"""
    
    # Benchmark generación
    tiempo_gen = timeit.timeit(
        lambda: generar_imagen_hueso_simulada(256, 256),
        number=10
    ) / 10
    
    # Benchmark análisis KS
    img1 = generar_imagen_hueso_simulada(256, 256, seed=1)
    img2 = generar_imagen_hueso_simulada(256, 256, seed=2)
    
    tiempo_ks = timeit.timeit(
        lambda: prueba_ks_imagenes(img1, img2),
        number=10
    ) / 10
    
    print("BENCHMARKS:")
    print(f"Generación (256x256): {tiempo_gen:.4f}s")
    print(f"Análisis KS (256x256): {tiempo_ks:.4f}s")
    print(f"Tiempo total: {tiempo_gen + tiempo_ks:.4f}s")
    
    # Throughput
    imagenes_por_segundo = 1 / (tiempo_gen + tiempo_ks)
    print(f"Throughput: {imagenes_por_segundo:.1f} análisis/segundo")

benchmark_completo()