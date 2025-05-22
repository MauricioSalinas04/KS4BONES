# Informe de Verificación de Preprocesamiento

Fecha: 2025-05-22 12:12:46

## Resumen de Calidad

## Normalization

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| min | 0.000 | No definido | ➖ |
| max | 255.000 | 255 | ✅ |
| mean | 30.236 | 50.00 - 200.00 | ❌ |
| std | 37.933 | 20.00 - 80.00 | ✅ |

## Equalization

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| contrast | 57.548 | 30.00 - 100.00 | ✅ |
| histogram_uniformity | 16558.074 | 100.00 - 2000.00 | ❌ |

## Enhancement

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| edge_strength | 0.007 | 10.00 - 50.00 | ❌ |
| texture_variance | 27.282 | 500.00 - 5000.00 | ❌ |

## Segmentation

| Métrica | Valor | Rango Esperado | Estado |
|---------|--------|----------------|--------|
| bone_coverage | 0.394 | 0.10 - 0.50 | ✅ |
| connected_components | 1.000 | 1.00 - 20.00 | ✅ |


## Score de Calidad Global

- Checks pasados: 5/9
- Score de calidad: 55.6%

## Imágenes Generadas

1. `1_normalized.png`: Imagen normalizada
2. `2_equalized.png`: Imagen ecualizada
3. `3_enhanced.png`: Realce trabecular
4. `4_bone_mask.png`: Máscara ósea
5. `5_final.png`: Imagen final procesada
