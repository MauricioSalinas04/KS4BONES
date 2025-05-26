import numpy as np
from typing import Tuple, Dict, Optional
from enum import Enum

class TipoTejido(Enum):
    """Enumeration para los tipos de tejido óseo."""
    NORMAL = "normal"
    OSTEOPENIA = "osteopenia"  
    OSTEOPOROSIS = "osteoporosis"

class ParametrosTejido:
    """Clase para almacenar parámetros específicos de cada tipo de tejido."""
    
    def __init__(self, densidad_base: float, rugosidad: float, 
                 conectividad: float, porosidad: float, 
                 intensidad_cortical: Tuple[int, int],
                 intensidad_trabecular: Tuple[int, int]):
        """
        Args:
            densidad_base: Densidad base del tejido (0.0-1.0)
            rugosidad: Nivel de rugosidad de la textura (0.0-1.0)
            conectividad: Conectividad trabecular (0.0-1.0)
            porosidad: Nivel de porosidad/espacios vacíos (0.0-1.0)
            intensidad_cortical: Rango de intensidades para hueso cortical (min, max)
            intensidad_trabecular: Rango de intensidades para hueso trabecular (min, max)
        """
        self.densidad_base = densidad_base
        self.rugosidad = rugosidad
        self.conectividad = conectividad
        self.porosidad = porosidad
        self.intensidad_cortical = intensidad_cortical
        self.intensidad_trabecular = intensidad_trabecular

def obtener_parametros_tejido(tipo: TipoTejido) -> ParametrosTejido:
    """
    Retorna los parámetros específicos para cada tipo de tejido.
    
    Args:
        tipo: Tipo de tejido óseo
        
    Returns:
        Objeto ParametrosTejido con los valores correspondientes
    """
    parametros = {
        TipoTejido.NORMAL: ParametrosTejido(
            densidad_base=0.85,
            rugosidad=0.3,
            conectividad=0.8,
            porosidad=0.2,
            intensidad_cortical=(180, 220),
            intensidad_trabecular=(120, 160)
        ),
        
        TipoTejido.OSTEOPENIA: ParametrosTejido(
            densidad_base=0.65,
            rugosidad=0.5,
            conectividad=0.6,
            porosidad=0.4,
            intensidad_cortical=(150, 190),
            intensidad_trabecular=(90, 130)
        ),
        
        TipoTejido.OSTEOPOROSIS: ParametrosTejido(
            densidad_base=0.4,
            rugosidad=0.7,
            conectividad=0.3,
            porosidad=0.6,
            intensidad_cortical=(120, 160),
            intensidad_trabecular=(60, 100)
        )
    }
    
    return parametros[tipo]

def generar_ruido_perlin_2d(width: int, height: int, scale: float = 0.1, 
                           octaves: int = 4, persistence: float = 0.5, 
                           lacunarity: float = 2.0, seed: Optional[int] = None) -> np.ndarray:
    """
    Genera ruido Perlin 2D simplificado para simular textura trabecular.
    
    Args:
        width, height: Dimensiones de la imagen
        scale: Escala del ruido (más pequeño = más detalle)
        octaves: Número de octavas (capas de detalle)
        persistence: Amplitud relativa entre octavas
        lacunarity: Frecuencia relativa entre octavas
        seed: Semilla para reproducibilidad
        
    Returns:
        Array 2D con valores de ruido normalizados [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Implementación simplificada de ruido Perlin usando interpolación
    noise = np.zeros((height, width))
    
    for octave in range(octaves):
        # Frecuencia y amplitud para esta octava
        freq = scale * (lacunarity ** octave)
        amp = persistence ** octave
        
        # Generar ruido base para esta octava
        x_coords = np.arange(width) * freq
        y_coords = np.arange(height) * freq
        
        # Crear grid de coordenadas
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Generar ruido usando funciones trigonométricas
        octave_noise = (np.sin(xx) * np.cos(yy) + 
                       np.sin(xx * 2.1) * np.cos(yy * 1.7) +
                       np.sin(xx * 0.8) * np.cos(yy * 2.3)) * amp
        
        noise += octave_noise
    
    # Normalizar a [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    return noise

def generar_patron_trabecular(width: int, height: int, 
                             parametros: ParametrosTejido,
                             seed: Optional[int] = None) -> np.ndarray:
    """
    Genera un patrón trabecular basado en los parámetros del tejido.
    
    Args:
        width, height: Dimensiones de la imagen
        parametros: Parámetros específicos del tipo de tejido
        seed: Semilla para reproducibilidad
        
    Returns:
        Array 2D representando el patrón trabecular
    """
    # Generar textura base con ruido Perlin
    textura_base = generar_ruido_perlin_2d(
        width, height, 
        scale=0.05 + parametros.rugosidad * 0.1,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        seed=seed
    )
    
    # Aplicar densidad base
    patron = textura_base * parametros.densidad_base
    
    # Simular conectividad trabecular
    conectividad_mask = generar_ruido_perlin_2d(
        width, height, 
        scale=0.02,
        octaves=2,
        seed=seed + 1 if seed else None
    )
    
    # Áreas bien conectadas mantienen más densidad
    patron = patron * (0.5 + 0.5 * conectividad_mask * parametros.conectividad)
    
    # Simular porosidad (espacios vacíos)
    porosidad_mask = generar_ruido_perlin_2d(
        width, height,
        scale=0.08,
        octaves=3,
        seed=seed + 2 if seed else None
    )
    
    # Crear poros donde la máscara supera el umbral
    umbral_poros = 1.0 - parametros.porosidad
    patron = np.where(porosidad_mask > umbral_poros, patron * 0.1, patron)
    
    return patron

def generar_imagen_hueso_simulada(width: int = 256, height: int = 256,
                                 tipo: TipoTejido = TipoTejido.NORMAL,
                                 incluir_cortical: bool = True,
                                 grosor_cortical: int = 15,
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    Genera una imagen simulada de tejido óseo con características específicas.
    
    Args:
        width, height: Dimensiones de la imagen
        tipo: Tipo de tejido óseo a simular
        incluir_cortical: Si incluir hueso cortical en los bordes
        grosor_cortical: Grosor del hueso cortical en píxeles
        seed: Semilla para reproducibilidad
        
    Returns:
        Array 2D con la imagen simulada en escala de grises [0, 255]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Obtener parámetros para el tipo de tejido
    parametros = obtener_parametros_tejido(tipo)
    
    # Generar patrón trabecular
    patron_trabecular = generar_patron_trabecular(width, height, parametros, seed)
    
    # Convertir a intensidades de escala de grises
    min_int, max_int = parametros.intensidad_trabecular
    imagen = (patron_trabecular * (max_int - min_int) + min_int).astype(np.uint8)
    
    # Añadir hueso cortical si se requiere
    if incluir_cortical:
        # Crear máscara para el hueso cortical (bordes)
        cortical_mask = np.zeros((height, width), dtype=bool)
        
        # Bordes externos
        cortical_mask[:grosor_cortical, :] = True  # Superior
        cortical_mask[-grosor_cortical:, :] = True  # Inferior
        cortical_mask[:, :grosor_cortical] = True  # Izquierdo
        cortical_mask[:, -grosor_cortical:] = True  # Derecho
        
        # Generar intensidades para hueso cortical
        min_cort, max_cort = parametros.intensidad_cortical
        intensidad_cortical = np.random.randint(min_cort, max_cort + 1, 
                                              size=cortical_mask.sum())
        
        # Aplicar hueso cortical
        imagen[cortical_mask] = intensidad_cortical
    
    # Añadir ruido realista
    ruido = np.random.normal(0, 5, (height, width))
    imagen = np.clip(imagen.astype(float) + ruido, 0, 255).astype(np.uint8)
    
    return imagen

def generar_dataset_simulado(n_imagenes_por_tipo: int = 50,
                           dimensiones: Tuple[int, int] = (256, 256),
                           incluir_cortical: bool = True,
                           seed_base: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Genera un dataset completo de imágenes simuladas para los tres tipos de tejido.
    
    Args:
        n_imagenes_por_tipo: Número de imágenes a generar por cada tipo
        dimensiones: Tupla (width, height) para el tamaño de las imágenes
        incluir_cortical: Si incluir hueso cortical
        seed_base: Semilla base para reproducibilidad
        
    Returns:
        Diccionario con arrays de imágenes por tipo
    """
    dataset = {}
    width, height = dimensiones
    
    for tipo in TipoTejido:
        print(f"Generando {n_imagenes_por_tipo} imágenes de tipo {tipo.value}...")
        
        imagenes = []
        for i in range(n_imagenes_por_tipo):
            # Usar semilla diferente para cada imagen
            semilla = (seed_base + i) if seed_base is not None else None
            
            imagen = generar_imagen_hueso_simulada(
                width=width,
                height=height,
                tipo=tipo,
                incluir_cortical=incluir_cortical,
                seed=semilla+i
            )
            
            imagenes.append(imagen)
        
        dataset[tipo.value] = np.array(imagenes)
    
    return dataset

def visualizar_comparacion_tipos(seed: int = 42) -> None:
    """
    Genera y muestra ejemplos de los tres tipos de tejido para comparación.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, tipo in enumerate(TipoTejido):
        imagen = generar_imagen_hueso_simulada(
            width=256, height=256, 
            tipo=tipo, 
            seed=seed + i
        )
        
        axes[i].imshow(imagen, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Tejido {tipo.value.capitalize()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
def ejemplo_uso():
    """Ejemplo de cómo usar el simulador."""
    
    print("=== SIMULADOR DE IMÁGENES DE OSTEOPOROSIS ===\n")
    
    # Generar una imagen de cada tipo
    for tipo in TipoTejido:
        imagen = generar_imagen_hueso_simulada(
            width=128, height=128,
            tipo=tipo,
            seed=42
        )
        
        print(f"Imagen {tipo.value}:")
        print(f"  - Dimensiones: {imagen.shape}")
        print(f"  - Rango intensidades: [{imagen.min()}, {imagen.max()}]")
        print(f"  - Intensidad promedio: {imagen.mean():.1f}")
        print()
    
    # Generar dataset pequeño
    print("Generando dataset de prueba...")
    dataset = generar_dataset_simulado(
        n_imagenes_por_tipo=5,
        dimensiones=(64, 64),
        seed_base=123
    )
    
    for tipo, imagenes in dataset.items():
        print(f"Dataset {tipo}: {imagenes.shape}")

if __name__ == "__main__":
    ejemplo_uso()
