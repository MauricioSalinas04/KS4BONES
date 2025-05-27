import sys
import os

# Agregar el directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importar módulos necesarios
import numpy as np
from generadorTejidos import TipoTejido, generar_imagen_hueso_simulada
from ks import prueba_ks_imagenes

def test_integracion_completa():
    """Prueba de integración del sistema completo"""
    
    print("Iniciando pruebas de integración...")
    
    # Generar datos de prueba
    tipos_tejido = [TipoTejido.NORMAL, TipoTejido.OSTEOPENIA, TipoTejido.OSTEOPOROSIS]
    imagenes_test = {}
    
    for tipo in tipos_tejido:
        imagenes_test[tipo] = generar_imagen_hueso_simulada(
            width=64, height=64,
            tipo=tipo,
            seed=42
        )
    
    # Test 1: Flujo completo de análisis
    resultado = prueba_ks_imagenes(
        imagenes_test[TipoTejido.NORMAL],
        imagenes_test[TipoTejido.OSTEOPOROSIS]
    )
    print(f"Test 1 (Normal vs Osteoporosis):")
    print(f"D = {resultado['estadistico_D']:.6f}")
    print(f"p-valor = {resultado['p_valor']:.6f}\n")
    
    # Test 2: Validar coherencia de resultados
    resultado_similar = prueba_ks_imagenes(
        imagenes_test[TipoTejido.NORMAL],
        imagenes_test[TipoTejido.NORMAL]
    )
    print(f"Test 2.1 (Normal vs Normal):")
    print(f"D = {resultado_similar['estadistico_D']:.6f}")
    print(f"p-valor = {resultado_similar['p_valor']:.6f}\n")
    
    resultado_diferente = prueba_ks_imagenes(
        imagenes_test[TipoTejido.NORMAL],
        imagenes_test[TipoTejido.OSTEOPOROSIS]
    )
    print(f"Test 2.2 (Normal vs Osteoporosis):")
    print(f"D = {resultado_diferente['estadistico_D']:.6f}")
    print(f"p-valor = {resultado_diferente['p_valor']:.6f}\n")
    
    # Valores esperados para tejidos similares
    assert resultado_similar['estadistico_D'] < 0.3, \
        "El estadístico D debería ser bajo para tejidos similares"
    
    # La comparación de tejidos iguales no debería rechazar H0
    assert not resultado_similar['rechazar_H0'], \
        "No debería rechazar H0 para el mismo tipo de tejido"
    
    print("✓ Pruebas de integración completadas exitosamente")

test_integracion_completa()