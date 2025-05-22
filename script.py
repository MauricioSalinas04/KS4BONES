import numpy as np
import cv2
from scipy import stats
from scipy.stats import ks_2samp
from skimage import filters, morphology, feature, measure, exposure
from skimage.filters import threshold_otsu, frangi, sobel_h, sobel_v
from skimage.morphology import skeletonize
from skimage.feature import graycomatrix, graycoprops
from skimage.exposure import equalize_adapthist
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

class HybridOsteoporosisDetector:
    """
    Detector híbrido de osteoporosis que combina análisis KS con características estructurales
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Inicializa el detector con pesos personalizables
        
        Args:
            weights: Diccionario con pesos para cada componente del análisis
        """
        self.weights = weights or {
            'density_global': 0.3,
            'density_regional': 0.2,
            'structural': 0.5
        }
        
        # Validar que los pesos sumen 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0, actual: {total_weight}")
    
    def verify_preprocessing(self, image: np.ndarray, output_dir: str) -> Dict:
        """
        Verifica y exporta cada fase del preprocesamiento

        Args:
            image: Imagen original en escala de grises
            output_dir: Directorio donde guardar las imágenes

        Returns:
            Dict con métricas de calidad de cada fase
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics = {}

        # 1. Normalización de intensidades
        normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, '1_normalized.png'), normalized)
        metrics['normalization'] = {
            'min': float(normalized.min()),
            'max': float(normalized.max()),
            'mean': float(normalized.mean()),
            'std': float(normalized.std())
        }

        # 2. Ecualización adaptativa
        equalized = equalize_adapthist(normalized, clip_limit=0.03)
        equalized = (equalized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, '2_equalized.png'), equalized)
        metrics['equalization'] = {
            'contrast': float(np.std(equalized)),
            'histogram_uniformity': float(np.histogram(equalized, bins=256)[0].std())
        }

        # 3. Realce trabecular
        enhanced = self.enhance_trabecular_pattern(equalized)
        cv2.imwrite(os.path.join(output_dir, '3_enhanced.png'), enhanced)
        metrics['enhancement'] = {
            'edge_strength': float(np.mean(filters.sobel(enhanced))),
            'texture_variance': float(np.var(enhanced))
        }

        # 4. Segmentación ósea
        bone_mask = self.segment_bone_tissue(equalized)
        cv2.imwrite(os.path.join(output_dir, '4_bone_mask.png'), bone_mask * 255)
        metrics['segmentation'] = {
            'bone_coverage': float(np.mean(bone_mask)),
            'connected_components': int(measure.label(bone_mask).max())
        }

        # Imagen final procesada
        final = enhanced * bone_mask
        cv2.imwrite(os.path.join(output_dir, '5_final.png'), final)

        # Agregar rangos aceptables para verificación
        metrics['quality_thresholds'] = {
            'normalization': {
                'min': 0,
                'max': 255,
                'mean': (50, 200),
                'std': (20, 80)
            },
            'equalization': {
                'contrast': (30, 100),
                'histogram_uniformity': (100, 2000)
            },
            'enhancement': {
                'edge_strength': (10, 50),
                'texture_variance': (500, 5000)
            },
            'segmentation': {
                'bone_coverage': (0.1, 0.5),
                'connected_components': (1, 20)
            }
        }

        return metrics

    # Modificar el método preprocess_xray para incluir verificación opcional
    def preprocess_xray(self, image: np.ndarray, verify: bool = False, output_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-procesamiento específico para análisis óseo con verificación opcional
        
        Args:
            image: Imagen de entrada en escala de grises
            verify: Si es True, realiza verificación y exporta resultados
            output_dir: Directorio para guardar resultados de verificación
            
        Returns:
            Tuple con (imagen procesada, máscara de hueso)
        """
        if verify:
            if output_dir is None:
                output_dir = 'preprocessing_verification'
            metrics = self.verify_preprocessing(image, output_dir)
            print("Métricas de preprocesamiento guardadas en:", output_dir)
            print(json.dumps(metrics, indent=2))

        # Normalizar intensidades
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        
        # 1. Ecualización adaptativa del histograma
        image_eq = equalize_adapthist(image, clip_limit=0.03)
        image_eq = (image_eq * 255).astype(np.uint8)
        
        # 2. Realce de estructura trabecular
        enhanced = self.enhance_trabecular_pattern(image_eq)
        
        # 3. Segmentación básica de hueso
        bone_mask = self.segment_bone_tissue(image_eq)
        
        # 4. Aplicar máscara
        processed_image = enhanced * bone_mask
        
        return processed_image, bone_mask
    
    def enhance_trabecular_pattern(self, image: np.ndarray) -> np.ndarray:
        """
        Realza el patrón trabecular usando filtros multi-escala
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Imagen con estructura trabecular realzada
        """
        # 1. Normalizar a rango [0,1]
        image_normalized = image.astype(np.float64) / 255.0
        
        # 2. Mejora de contraste local
        enhanced = exposure.equalize_adapthist(image_normalized, clip_limit=0.03)
        
        # 3. Filtro Frangi modificado
        vessel_like = np.zeros_like(enhanced)
        for sigma in [1.0, 2.0, 3.0]:
            try:
                frangi_filtered = frangi(
                    enhanced,
                    sigmas=[sigma],
                    beta=0.5,
                    gamma=15,
                    black_ridges=False
                )
                vessel_like += frangi_filtered
            except:
                continue
                
        # 4. Normalización y ajuste de contraste
        if vessel_like.max() > vessel_like.min():
            # Normalizar a [0,255]
            enhanced = ((vessel_like - vessel_like.min()) / 
                       (vessel_like.max() - vessel_like.min()) * 255)
            enhanced = enhanced.astype(np.uint8)
            
            # Mejora adicional de contraste
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=10)
        else:
            # Si el realce falla, devolver imagen original mejorada
            enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        
        # 5. Filtro de suavizado para reducir ruido
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0.5)
        
        return enhanced
    
    def segment_bone_tissue(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentación básica para separar hueso de tejido blando
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Máscara binaria del hueso
        """
        # Threshold adaptativo
        threshold = threshold_otsu(image)
        bone_mask = image > threshold * 0.7  # Más permisivo para capturar hueso trabecular
        
        # Operaciones morfológicas para limpiar
        bone_mask = morphology.remove_small_objects(bone_mask, min_size=100)
        bone_mask = morphology.binary_closing(bone_mask, morphology.disk(2))
        
        return bone_mask.astype(np.uint8)
    
    def multi_regional_ks_analysis(self, image1: np.ndarray, image2: np.ndarray, 
                                 grid_size: Tuple[int, int] = (4, 4)) -> List[Dict]:
        """
        Análisis KS multi-regional dividiendo las imágenes en sub-regiones
        
        Args:
            image1: Primera imagen (normal)
            image2: Segunda imagen (test)
            grid_size: Tamaño de la grilla para dividir las imágenes
            
        Returns:
            Lista de resultados por región
        """
        results = []
        h, w = image1.shape
        step_h, step_w = h // grid_size[0], w // grid_size[1]
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Extraer sub-región
                start_h, end_h = i * step_h, (i + 1) * step_h
                start_w, end_w = j * step_w, (j + 1) * step_w
                
                roi1 = image1[start_h:end_h, start_w:end_w]
                roi2 = image2[start_h:end_h, start_w:end_w]
                
                # Verificar que hay suficientes píxeles de hueso
                roi1_bone = roi1[roi1 > 0]
                roi2_bone = roi2[roi2 > 0]
                
                if len(roi1_bone) > 50 and len(roi2_bone) > 50:
                    try:
                        ks_stat, p_value = ks_2samp(roi1_bone, roi2_bone)
                        
                        results.append({
                            'region': (i, j),
                            'coordinates': (start_h, end_h, start_w, end_w),
                            'ks_distance': ks_stat,
                            'p_value': p_value,
                            'bone_density_1': np.mean(roi1_bone),
                            'bone_density_2': np.mean(roi2_bone),
                            'bone_pixels_1': len(roi1_bone),
                            'bone_pixels_2': len(roi2_bone)
                        })
                    except:
                        # Si KS falla, agregar región con valores por defecto
                        results.append({
                            'region': (i, j),
                            'coordinates': (start_h, end_h, start_w, end_w),
                            'ks_distance': 0.0,
                            'p_value': 1.0,
                            'bone_density_1': 0.0,
                            'bone_density_2': 0.0,
                            'bone_pixels_1': 0,
                            'bone_pixels_2': 0
                        })
        
        return results
    
    def structural_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extrae características estructurales del hueso
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Diccionario con características estructurales
        """
        features = {}
        
        try:
            # 1. ANÁLISIS DE TEXTURA (GLCM)
            # Reducir imagen para GLCM (computacionalmente costoso)
            if image.shape[0] > 512 or image.shape[1] > 512:
                scale_factor = min(512 / image.shape[0], 512 / image.shape[1])
                new_height = int(image.shape[0] * scale_factor)
                new_width = int(image.shape[1] * scale_factor)
                image_resized = cv2.resize(image, (new_width, new_height))
            else:
                image_resized = image.copy()
            
            # Cuantizar a 8 niveles para GLCM
            image_quantized = (image_resized // 32).astype(np.uint8)
            
            glcm = graycomatrix(image_quantized, distances=[1, 2], 
                              angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                              levels=8, symmetric=True, normed=True)
            
            features['contrast'] = graycoprops(glcm, 'contrast').mean()
            features['homogeneity'] = graycoprops(glcm, 'homogeneity').mean()
            features['energy'] = graycoprops(glcm, 'energy').mean()
            features['correlation'] = graycoprops(glcm, 'correlation').mean()
            
        except Exception as e:
            # Valores por defecto si GLCM falla
            features.update({
                'contrast': 0.0,
                'homogeneity': 0.5,
                'energy': 0.0,
                'correlation': 0.0
            })
        
        try:
            # 2. ANÁLISIS FRACTAL
            features['fractal_dimension'] = self.calculate_fractal_dimension(image)
        except:
            features['fractal_dimension'] = 2.0
        
        try:
            # 3. ANÁLISIS DE CONECTIVIDAD
            binary_bone = image > threshold_otsu(image)
            if np.sum(binary_bone) > 100:
                skeleton = skeletonize(binary_bone)
                features['skeleton_density'] = np.sum(skeleton) / max(np.sum(binary_bone), 1)
            else:
                features['skeleton_density'] = 0.0
        except:
            features['skeleton_density'] = 0.0
        
        try:
            # 4. ANISOTROPÍA DIRECCIONAL
            features['anisotropy'] = self.calculate_directional_anisotropy(image)
        except:
            features['anisotropy'] = 0.0
        
        try:
            # 5. HETEROGENEIDAD ESPACIAL
            features['spatial_heterogeneity'] = self.calculate_spatial_variance(image)
        except:
            features['spatial_heterogeneity'] = 0.0
        
        return features
    
    def calculate_fractal_dimension(self, image: np.ndarray) -> float:
        """
        Calcula la dimensión fractal usando box-counting
        
        Args:
            image: Imagen binaria o en escala de grises
            
        Returns:
            Dimensión fractal
        """
        # Convertir a binario si es necesario
        if len(np.unique(image)) > 2:
            binary_image = image > threshold_otsu(image)
        else:
            binary_image = image.astype(bool)
        
        # Tamaños de caja para box-counting
        box_sizes = [2, 4, 8, 16, 32, 64]
        counts = []
        
        for box_size in box_sizes:
            if box_size < min(binary_image.shape):
                # Contar cajas que contienen píxeles activos
                count = 0
                for i in range(0, binary_image.shape[0], box_size):
                    for j in range(0, binary_image.shape[1], box_size):
                        box = binary_image[i:i+box_size, j:j+box_size]
                        if np.any(box):
                            count += 1
                counts.append(count)
            else:
                counts.append(1)
        
        # Ajuste linear en escala log-log
        if len(counts) >= 2 and max(counts) > 0:
            log_boxes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(np.maximum(counts, 1))  # Evitar log(0)
            
            # Regresión linear
            slope, _ = np.polyfit(log_boxes, log_counts, 1)
            fractal_dim = -slope
            
            # Limitar a rango razonable
            fractal_dim = np.clip(fractal_dim, 1.0, 3.0)
        else:
            fractal_dim = 2.0
        
        return float(fractal_dim)
    
    def calculate_directional_anisotropy(self, image: np.ndarray) -> float:
        """
        Calcula la anisotropía direccional de las estructuras
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Índice de anisotropía (0=isotrópico, 1=altamente anisotrópico)
        """
        # Gradientes direccionales
        grad_h = sobel_h(image)
        grad_v = sobel_v(image)
        
        # Magnitud y dirección del gradiente
        magnitude = np.sqrt(grad_h**2 + grad_v**2)
        direction = np.arctan2(grad_v, grad_h)
        
        # Solo considerar gradientes significativos
        threshold = np.percentile(magnitude, 75)
        mask = magnitude > threshold
        
        if np.sum(mask) > 10:
            # Histograma de direcciones ponderado por magnitud
            hist, _ = np.histogram(direction[mask], bins=36, 
                                 weights=magnitude[mask], density=True)
            
            # Índice de anisotropía
            anisotropy = (np.max(hist) - np.min(hist)) / np.max(hist) if np.max(hist) > 0 else 0.0
        else:
            anisotropy = 0.0
        
        return float(anisotropy)
    
    def calculate_spatial_variance(self, image: np.ndarray) -> float:
        """
        Calcula la heterogeneidad espacial de la imagen
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            Índice de heterogeneidad espacial
        """
        # Dividir imagen en bloques pequeños
        block_size = 32
        h, w = image.shape
        block_means = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    block_means.append(np.mean(block))
        
        # Varianza de las medias de los bloques
        if len(block_means) > 1:
            spatial_variance = np.var(block_means) / np.mean(block_means) if np.mean(block_means) > 0 else 0.0
        else:
            spatial_variance = 0.0
        
        return float(spatial_variance)
    
    def analyze_pair(self, normal_xray: np.ndarray, test_xray: np.ndarray) -> Dict:
        """
        Análisis completo de un par de radiografías
        
        Args:
            normal_xray: Radiografía normal de referencia
            test_xray: Radiografía a evaluar
            
        Returns:
            Diccionario con resultados completos del análisis
        """
        # Pre-procesamiento
        normal_processed, normal_mask = self.preprocess_xray(normal_xray)
        test_processed, test_mask = self.preprocess_xray(test_xray)
        
        results = {
            'preprocessing': {
                'normal_bone_pixels': int(np.sum(normal_mask)),
                'test_bone_pixels': int(np.sum(test_mask)),
                'bone_coverage_normal': float(np.sum(normal_mask) / normal_mask.size),
                'bone_coverage_test': float(np.sum(test_mask) / test_mask.size)
            }
        }
        
        # NIVEL 1: Análisis KS Global
        try:
            normal_bone_pixels = normal_processed[normal_processed > 0]
            test_bone_pixels = test_processed[test_processed > 0]
            
            if len(normal_bone_pixels) > 0 and len(test_bone_pixels) > 0:
                ks_global, p_global = ks_2samp(normal_bone_pixels, test_bone_pixels)
            else:
                ks_global, p_global = 0.0, 1.0
            
            results['global_ks'] = {
                'distance': float(ks_global),
                'p_value': float(p_global),
                'significant': p_global < 0.05,
                'mean_density_normal': float(np.mean(normal_bone_pixels)) if len(normal_bone_pixels) > 0 else 0.0,
                'mean_density_test': float(np.mean(test_bone_pixels)) if len(test_bone_pixels) > 0 else 0.0
            }
        except Exception as e:
            results['global_ks'] = {
                'distance': 0.0,
                'p_value': 1.0,
                'significant': False,
                'error': str(e)
            }
        
        # NIVEL 2: Análisis KS Regional
        try:
            regional_results = self.multi_regional_ks_analysis(normal_processed, test_processed)
            
            if regional_results:
                significant_regions = [r for r in regional_results if r['p_value'] < 0.05]
                ks_distances = [r['ks_distance'] for r in regional_results]
                
                results['regional_ks'] = {
                    'regions_analyzed': len(regional_results),
                    'regions_significant': len(significant_regions),
                    'max_distance': float(max(ks_distances)) if ks_distances else 0.0,
                    'mean_distance': float(np.mean(ks_distances)) if ks_distances else 0.0,
                    'heterogeneity_score': float(np.std(ks_distances)) if len(ks_distances) > 1 else 0.0,
                    'significant_regions_coords': [r['coordinates'] for r in significant_regions]
                }
            else:
                results['regional_ks'] = {
                    'regions_analyzed': 0,
                    'regions_significant': 0,
                    'max_distance': 0.0,
                    'mean_distance': 0.0,
                    'heterogeneity_score': 0.0,
                    'significant_regions_coords': []
                }
        except Exception as e:
            results['regional_ks'] = {
                'error': str(e),
                'regions_analyzed': 0,
                'regions_significant': 0,
                'max_distance': 0.0,
                'heterogeneity_score': 0.0
            }
        
        # NIVEL 3: Análisis Estructural
        try:
            struct_normal = self.structural_analysis(normal_processed)
            struct_test = self.structural_analysis(test_processed)
            
            structural_changes = {}
            for key in struct_normal:
                if struct_normal[key] != 0:
                    change = abs(struct_normal[key] - struct_test[key]) / abs(struct_normal[key])
                else:
                    change = 0.0
                structural_changes[f'{key}_change'] = float(change)
            
            results['structural'] = {
                'normal_features': struct_normal,
                'test_features': struct_test,
                'changes': structural_changes
            }
        except Exception as e:
            results['structural'] = {
                'error': str(e),
                'changes': {}
            }
        
        # NIVEL 4: Score Compuesto
        try:
            composite_result = self.calculate_composite_score(results)
            results['composite_score'] = composite_result
        except Exception as e:
            results['composite_score'] = {
                'score': 0.0,
                'interpretation': 'Error en el cálculo',
                'error': str(e)
            }
        
        return results
    
    def calculate_composite_score(self, results: Dict) -> Dict:
        """
        Calcula el score compuesto combinando todos los análisis
        
        Args:
            results: Resultados de todos los análisis
            
        Returns:
            Diccionario con score compuesto e interpretación
        """
        # Score de densidad global (0-1)
        global_ks = results.get('global_ks', {})
        density_global_score = min(global_ks.get('distance', 0) * 2, 1.0)
        
        # Score de heterogeneidad regional (0-1)
        regional_ks = results.get('regional_ks', {})
        regional_score = min(regional_ks.get('heterogeneity_score', 0), 1.0)
        
        # Score estructural compuesto (0-1)
        structural_changes = results.get('structural', {}).get('changes', {})
        key_structural_features = [
            'contrast_change', 
            'fractal_dimension_change',
            'skeleton_density_change', 
            'anisotropy_change'
        ]
        
        structural_scores = []
        for feature in key_structural_features:
            score = structural_changes.get(feature, 0)
            structural_scores.append(min(score, 1.0))  # Limitar a 1.0
        
        structural_score = np.mean(structural_scores) if structural_scores else 0.0
        
        # Score final ponderado
        composite = (
            self.weights['density_global'] * density_global_score +
            self.weights['density_regional'] * regional_score +
            self.weights['structural'] * structural_score
        )
        
        # Calcular confianza basada en disponibilidad de datos
        confidence_factors = []
        if global_ks.get('distance', 0) > 0:
            confidence_factors.append(0.4)
        if regional_ks.get('regions_analyzed', 0) > 0:
            confidence_factors.append(0.3)
        if len(structural_changes) > 0:
            confidence_factors.append(0.3)
        
        confidence = sum(confidence_factors)
        
        return {
            'score': float(composite),
            'confidence': float(confidence),
            'components': {
                'density_global': float(density_global_score),
                'density_regional': float(regional_score),
                'structural': float(structural_score)
            },
            'interpretation': self.interpret_score(composite),
            'recommendation': self.generate_recommendation(composite, confidence)
        }
    
    def interpret_score(self, score: float) -> str:
        """
        Interpretación clínica del score compuesto
        
        Args:
            score: Score compuesto (0-1)
            
        Returns:
            Interpretación textual
        """
        if score < 0.2:
            return "Normal - No evidencia radiográfica de osteoporosis"
        elif score < 0.4:
            return "Leve - Posibles cambios compatibles con osteopenia"
        elif score < 0.7:
            return "Moderado - Sugestivo de osteoporosis"
        else:
            return "Severo - Altamente sugestivo de osteoporosis"
    
    def generate_recommendation(self, score: float, confidence: float) -> str:
        """
        Genera recomendación clínica basada en score y confianza
        
        Args:
            score: Score compuesto
            confidence: Nivel de confianza
            
        Returns:
            Recomendación clínica
        """
        if confidence < 0.5:
            return "Calidad de imagen limitada. Repetir estudio o considerar método alternativo."
        
        if score < 0.2:
            return "Seguimiento rutinario según protocolo institucional."
        elif score < 0.4:
            return "Considerar DXA para evaluación cuantitativa de densidad ósea."
        elif score < 0.7:
            return "Recomendada evaluación con DXA. Considerar factores de riesgo adicionales."
        else:
            return "Evaluación urgente con DXA recomendada. Considerar inicio de tratamiento según protocolo clínico."
    
    def generate_report(self, results: Dict, normal_image_path: str = "", 
                   test_image_path: str = "", patient_info: Dict = None) -> Dict:
        """
        Genera reporte completo en formato estandarizado
        
        Args:
            results: Resultados del análisis
            normal_image_path: Ruta de la imagen normal
            test_image_path: Ruta de la imagen de prueba
            patient_info: Información del paciente (opcional)
            
        Returns:
            Reporte completo en formato JSON
        """
        # Función auxiliar para convertir tipos no serializables
        def convert_to_serializable(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
    
        # Convertir resultados a formato serializable
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_results[key] = convert_to_serializable(value)
    
        report = {
            "analysis_id": f"OST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "images": {
                "normal": normal_image_path,
                "test": test_image_path
            },
            "patient_info": patient_info or {},
            "analysis_parameters": {
                "weights": self.weights,
                "preprocessing": "Adaptive histogram equalization + trabecular enhancement"
            },
            "results": serializable_results,
            "summary": {
                "final_score": float(serializable_results.get('composite_score', {}).get('score', 0.0)),
                "confidence": float(serializable_results.get('composite_score', {}).get('confidence', 0.0)),
                "interpretation": str(serializable_results.get('composite_score', {}).get('interpretation', 'Unknown')),
                "recommendation": str(serializable_results.get('composite_score', {}).get('recommendation', 'Consult specialist'))
            },
            "quality_metrics": {
                "normal_bone_coverage": float(serializable_results.get('preprocessing', {}).get('bone_coverage_normal', 0.0)),
                "test_bone_coverage": float(serializable_results.get('preprocessing', {}).get('bone_coverage_test', 0.0)),
                "regions_analyzed": int(serializable_results.get('regional_ks', {}).get('regions_analyzed', 0)),
                "structural_features_extracted": len(serializable_results.get('structural', {}).get('changes', {}))
            }
        }
        
        return report

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

# Ejemplo de uso
def main_example():
    """
    Ejemplo de uso del detector de osteoporosis
    """
    # Inicializar detector
    detector = HybridOsteoporosisDetector()
    
    try:
        # Cargar imágenes
        normal_image = load_image("data/normal_xray.jpg")
        test_image = load_image("data/test_xray.jpg")

        # Verificar preprocesamiento
        normal_processed, normal_mask = detector.preprocess_xray(
            normal_image, 
            verify=True, 
            output_dir="verification/normal"
        )
        
        test_processed, test_mask = detector.preprocess_xray(
            test_image,
            verify=True,
            output_dir="verification/test"
        )

        # Información del paciente de ejemplo
        patient_info = {
            "id": "P12345",
            "age": 65,
            "gender": "F",
            "study_date": datetime.now().strftime("%Y-%m-%d")
        }

        # Realizar análisis
        results = detector.analyze_pair(normal_image, test_image)

        # Generar reporte
        report = detector.generate_report(
            results, 
            normal_image_path="data/normal_xray.jpg",
            test_image_path="data/test_xray.jpg",
            patient_info=patient_info
        )

        # Guardar reporte
        save_report(report, "results/analysis_report.json")

        # Visualizar resultados
        visualize_results(normal_image, test_image, results, "results/analysis_visualization.png")

        print("Análisis completado exitosamente")
        print(f"Score final: {results['composite_score']['score']:.3f}")
        print(f"Interpretación: {results['composite_score']['interpretation']}")

    except Exception as e:
        print(f"Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    main_example()
