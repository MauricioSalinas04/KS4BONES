# Informe de Verificación de Preprocesamiento

Fecha: 2025-05-22 14:56:37

## Resumen de Calidad

## Normalization

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| min | 0.000 | 0 | ✅ |
| max | 255.000 | 255 | ✅ |
| mean | 56.086 | 50.00 - 200.00 | ✅ |
| std | 50.502 | 20.00 - 80.00 | ✅ |

## Equalization

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| contrast | 56.835 | 30.00 - 100.00 | ✅ |
| histogram_uniformity | 5264.892 | 100.00 - 2000.00 | ❌ |

## Enhancement

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| edge_strength | 0.058 | 10.00 - 50.00 | ❌ |
| texture_variance | 726.353 | 500.00 - 5000.00 | ✅ |

## Segmentation

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| bone_coverage | 0.252 | 0.10 - 0.50 | ✅ |
| connected_components | 2.000 | 1.00 - 20.00 | ✅ |


## Score de Calidad Global

- Checks pasados: 8/10
- Score de calidad: 80.0%

## Imágenes Generadas

1. `1_normalized.png`: Imagen normalizada
2. `2_equalized.png`: Imagen ecualizada
3. `3_enhanced.png`: Realce trabecular
4. `4_bone_mask.png`: Máscara ósea
5. `5_final.png`: Imagen final procesada
