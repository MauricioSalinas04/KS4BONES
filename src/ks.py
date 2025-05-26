import numpy as np
import cv2

def calcular_histograma_normalizado(imagen: np.ndarray) -> np.ndarray:
    """
    Calcula el histograma normalizado de una imagen en escala de grises.
    
    Args:
        imagen: Array numpy de la imagen en escala de grises
        
    Returns:
        Array con el histograma normalizado (probabilidades)
    """
    # Calcular histograma para niveles 0-255
    hist, _ = np.histogram(imagen.flatten(), bins=256, range=(0, 256))
    
    # Normalizar para obtener probabilidades
    hist_normalizado = hist / np.sum(hist)
    
    return hist_normalizado

def calcular_fda_empirica(histograma_normalizado: np.ndarray) -> np.ndarray:
    """
    Calcula la función de distribución acumulada (FDA) empírica.
    
    Args:
        histograma_normalizado: Histograma normalizado de la imagen
        
    Returns:
        Array con la FDA empírica F_g
    """
    # F_g = sum(h_g' para g'=0 hasta g)
    fda = np.cumsum(histograma_normalizado)
    
    return fda

def calcular_estadistico_ks(fda1: np.ndarray, fda2: np.ndarray) -> float:
    """
    Calcula el estadístico D^ de Kolmogorov-Smirnov.
    
    Args:
        fda1: FDA de la primera imagen
        fda2: FDA de la segunda imagen
        
    Returns:
        Valor del estadístico D^
    """
    # D^ = max_g |F_g^(1) - F_g^(2)|
    diferencias = np.abs(fda1 - fda2)
    d_hat = np.max(diferencias)
    
    return d_hat

def calcular_lambda_ks(d_hat: float, p1: int, q1: int, p2: int, q2: int) -> float:
    """
    Calcula el parámetro λ_KS para la prueba KS.
    
    Args:
        d_hat: Estadístico D^ calculado
        p1, q1: Dimensiones de la primera imagen
        p2, q2: Dimensiones de la segunda imagen
        
    Returns:
        Valor de λ_KS
    """
    # J = (P1*Q1 * P2*Q2) / (P1*Q1 + P2*Q2)
    n1 = p1 * q1  # Número total de píxeles imagen 1
    n2 = p2 * q2  # Número total de píxeles imagen 2
    
    j = (n1 * n2) / (n1 + n2)
    
    # λ_KS = D^ * sqrt(J + 0.11/sqrt(J + 0.12))
    lambda_ks = d_hat * (np.sqrt(j) + (0.11 / np.sqrt(j)) + 0.12)
    
    return lambda_ks

def calcular_q_ks(lambda_val: float, max_terminos: int = 100) -> float:
    """
    Calcula el valor-p usando la función Q_KS.
    
    Args:
        lambda_val: Valor de λ_KS
        max_terminos: Número máximo de términos en la serie infinita
        
    Returns:
        Valor de Q_KS (valor-p)
    """
    # Q_KS(λ) = 2 * sum(j=1 to inf) (-1)^(j-1) * exp(-2*j^2*λ^2)
    q_ks = 0.0

    if lambda_val <= 0:
        return 1.0
    
    for j in range(1, max_terminos + 1):
        termino = ((-1) ** (j - 1)) * np.exp(-2 * j**2 * lambda_val**2)
        q_ks += termino
        
        # Criterio de convergencia para series infinitas
        if abs(termino) < 1e-10:
            break
    
    q_ks *= 2
    
    # Asegurar que esté en rango [0, 1]
    q_ks = max(0.0, min(1.0, q_ks))
    
    return q_ks

def prueba_ks_imagenes(imagen1: np.ndarray, imagen2: np.ndarray, 
                      alpha: float = 0.05) -> dict:
    """
    Realiza la prueba completa de Kolmogorov-Smirnov entre dos imágenes.
    
    Args:
        imagen1: Primera imagen en escala de grises
        imagen2: Segunda imagen en escala de grises
        alpha: Nivel de significancia (por defecto 0.05)
        
    Returns:
        Diccionario con los resultados de la prueba
    """
    # Convertir a escala de grises si es necesario
    if len(imagen1.shape) == 3:
        imagen1 = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    if len(imagen2.shape) == 3:
        imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    
    # Paso 1: Calcular histogramas normalizados
    hist1 = calcular_histograma_normalizado(imagen1)
    hist2 = calcular_histograma_normalizado(imagen2)
    
    # Paso 2: Calcular FDA empíricas
    fda1 = calcular_fda_empirica(hist1)
    fda2 = calcular_fda_empirica(hist2)
    
    # Paso 3: Calcular estadístico D^
    d_hat = calcular_estadistico_ks(fda1, fda2)
    
    # Paso 4: Calcular λ_KS
    p1, q1 = imagen1.shape
    p2, q2 = imagen2.shape
    lambda_ks = calcular_lambda_ks(d_hat, p1, q1, p2, q2)
    
    # Paso 5: Calcular valor-p (Q_KS)
    p_valor = calcular_q_ks(lambda_ks)
    print(f"Valor-p calculado: {p_valor:.6f}")
    
    # Paso 6: Decisión estadística
    rechazar_h0 = p_valor < alpha
    
    # Preparar resultados
    resultados = {
        'estadistico_D': d_hat,
        'lambda_KS': lambda_ks,
        'p_valor': p_valor,
        'alpha': alpha,
        'rechazar_H0': rechazar_h0,
        'conclusion': 'Las imágenes tienen distribuciones diferentes' if rechazar_h0 
                     else 'No hay evidencia suficiente de diferencia entre distribuciones',
        'histograma_1': hist1,
        'histograma_2': hist2,
        'fda_1': fda1,
        'fda_2': fda2
    }
    
    return resultados

# Ejemplo de uso
def ejemplo_uso():
    """
    Ejemplo de cómo usar la función de prueba KS.
    """
    # Simular dos imágenes diferentes
    np.random.seed(42)
    
    # Imagen 1: distribución normal
    imagen1 = np.random.normal(128, 40, (100, 100)).astype(np.uint8)
    imagen1 = np.clip(imagen1, 0, 255)
    
    # Imagen 2: distribución uniforme
    imagen2 = np.random.uniform(0, 255, (100, 100)).astype(np.uint8)
    
    # Realizar prueba KS
    resultado = prueba_ks_imagenes(imagen1, imagen2)
    
    # Mostrar resultados
    print("=== RESULTADOS PRUEBA KOLMOGOROV-SMIRNOV ===")
    print(f"Estadístico D^: {resultado['estadistico_D']:.6f}")
    print(f"Lambda KS: {resultado['lambda_KS']:.6f}")
    print(f"Valor-p: {resultado['p_valor']:.6f}")
    print(f"Nivel de significancia: {resultado['alpha']}")
    print(f"Rechazar H0: {resultado['rechazar_H0']}")
    print(f"Conclusión: {resultado['conclusion']}")

if __name__ == "__main__":
    ejemplo_uso()
