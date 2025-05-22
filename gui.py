import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from script import HybridOsteoporosisDetector, load_image
from datetime import datetime
import json

class OsteoporosisAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Osteoporosis")
        
        # Variables para almacenar rutas de imágenes
        self.normal_image_path = tk.StringVar()
        self.test_image_path = tk.StringVar()
        
        # Variables para almacenar imágenes
        self.normal_image = None
        self.test_image = None
        
        # Inicializar detector
        self.detector = HybridOsteoporosisDetector()
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Botones de carga
        ttk.Button(main_frame, text="Cargar Imagen 1", 
                  command=lambda: self.load_image(1)).grid(row=0, column=0, padx=5)
        ttk.Button(main_frame, text="Cargar Imagen 2",
                  command=lambda: self.load_image(2)).grid(row=0, column=1, padx=5)
        
        # Frame para visualización
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, pady=10)
        
        # Botones de análisis
        ttk.Button(main_frame, text="Ejecutar Test KS",
                  command=self.run_analysis).grid(row=2, column=0, padx=5)
        ttk.Button(main_frame, text="Guardar Resultados",
                  command=self.save_results).grid(row=2, column=1, padx=5)
        
        # Área de resultados
        self.results_text = tk.Text(main_frame, height=10, width=50)
        self.results_text.grid(row=3, column=0, columnspan=4, pady=10)
        
    def load_image(self, image_num):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        
        if file_path:
            try:
                if image_num == 1:
                    self.normal_image = load_image(file_path)
                    self.normal_image_path.set(file_path)
                else:
                    self.test_image = load_image(file_path)
                    self.test_image_path.set(file_path)
                
                self.update_display()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")
    
    def update_display(self):
        self.axes[0,0].clear()
        self.axes[0,1].clear()
        self.axes[1,0].clear()
        self.axes[1,1].clear()
        
        if self.normal_image is not None:
            self.axes[0,0].imshow(self.normal_image, cmap='gray')
            self.axes[0,0].set_title('Imagen Normal')
            self.axes[0,0].axis('off')
        
        if self.test_image is not None:
            self.axes[0,1].imshow(self.test_image, cmap='gray')
            self.axes[0,1].set_title('Imagen de Prueba')
            self.axes[0,1].axis('off')
            
        if self.normal_image is not None and self.test_image is not None:
            self.axes[1,0].hist(self.normal_image.flatten(), bins=50, 
                              alpha=0.7, label='Normal', density=True)
            self.axes[1,0].hist(self.test_image.flatten(), bins=50,
                              alpha=0.7, label='Prueba', density=True)
            self.axes[1,0].set_title('Histogramas de Intensidad')
            self.axes[1,0].legend()
        
        self.canvas.draw()
    
    def run_analysis(self):
        if self.normal_image is None or self.test_image is None:
            messagebox.showwarning("Advertencia", 
                                 "Por favor cargue ambas imágenes primero")
            return
        
        try:
            results = self.detector.analyze_pair(self.normal_image, self.test_image)
            self.current_results = results
            
            # Mostrar resultados en el área de texto
            composite_score = results.get('composite_score', {})
            report_text = f"""
            Score Compuesto: {composite_score.get('score', 0):.3f}
            Confianza: {composite_score.get('confidence', 0):.3f}
            
            Interpretación:
            {composite_score.get('interpretation', 'N/A')}
            
            KS Global: {results.get('global_ks', {}).get('distance', 0):.3f}
            Regiones Significativas: {results.get('regional_ks', {}).get('regions_significant', 0)}
            """
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, report_text)
            
            # Actualizar gráfico de resultados
            self.axes[1,1].clear()
            self.axes[1,1].text(0.1, 0.9, report_text, 
                              transform=self.axes[1,1].transAxes,
                              verticalalignment='top', fontsize=8,
                              fontfamily='monospace')
            self.axes[1,1].set_title('Resultados del Análisis')
            self.axes[1,1].axis('off')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el análisis: {str(e)}")
    
    def save_results(self):
        if not hasattr(self, 'current_results'):
            messagebox.showwarning("Advertencia",
                                 "Ejecute el análisis primero")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            
            if file_path:
                report = self.detector.generate_report(
                    self.current_results,
                    normal_image_path=self.normal_image_path.get(),
                    test_image_path=self.test_image_path.get()
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                    
                messagebox.showinfo("Éxito", 
                                  "Resultados guardados correctamente")
                
        except Exception as e:
            messagebox.showerror("Error",
                               f"Error al guardar resultados: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OsteoporosisAnalyzerGUI(root)
    root.mainloop()
    