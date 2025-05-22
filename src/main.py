from datetime import datetime
from HybridODetector import HybridOsteoporosisDetector
from utilities import load_image, save_report
from gui import visualize_results

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