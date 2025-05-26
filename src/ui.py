import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ks
import generadorTejidos as gt

# Importar las funciones que ya creamos (simulador y KS)
# (Aquí incluirías los imports de tus módulos anteriores)



class DetectorOsteoporosisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Osteoporosis - Análisis Kolmogorov-Smirnov")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables para almacenar imágenes
        self.imagen1 = None
        self.imagen2 = None
        self.resultados_ks = None
        
        self.crear_interfaz()
    
    def crear_interfaz(self):
        """Crea la interfaz principal con pestañas."""
        # Título principal
        titulo = tk.Label(self.root, text="Detector de Osteoporosis", 
                         font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#333')
        titulo.pack(pady=10)
        
        subtitulo = tk.Label(self.root, text="Análisis mediante Prueba Kolmogorov-Smirnov", 
                            font=("Arial", 12), bg='#f0f0f0', fg='#666')
        subtitulo.pack(pady=(0, 20))
        
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Crear pestañas
        self.crear_pestana_simulaciones()
        self.crear_pestana_analisis_ks()
    
    def crear_pestana_simulaciones(self):
        """Crea la pestaña de simulaciones."""
        frame_sim = ttk.Frame(self.notebook)
        self.notebook.add(frame_sim, text="Generador de Simulaciones")
        
        # Frame para controles
        frame_controles = tk.Frame(frame_sim, bg='white', relief='raised', bd=2)
        frame_controles.pack(fill='x', padx=10, pady=10)
        
        tk.Label(frame_controles, text="Parámetros de Simulación", 
                font=("Arial", 14, "bold"), bg='white').pack(pady=10)
        
        # Tipo de tejido
        tk.Label(frame_controles, text="Tipo de Tejido:", bg='white').pack(anchor='w', padx=20)
        self.tipo_tejido = tk.StringVar(value="normal")
        frame_tipo = tk.Frame(frame_controles, bg='white')
        frame_tipo.pack(fill='x', padx=20, pady=5)
        
        tk.Radiobutton(frame_tipo, text="Normal", variable=self.tipo_tejido, 
                      value="normal", bg='white').pack(side='left', padx=10)
        tk.Radiobutton(frame_tipo, text="Osteopenia", variable=self.tipo_tejido, 
                      value="osteopenia", bg='white').pack(side='left', padx=10)
        tk.Radiobutton(frame_tipo, text="Osteoporosis", variable=self.tipo_tejido, 
                      value="osteoporosis", bg='white').pack(side='left', padx=10)
        
        # Dimensiones
        frame_dim = tk.Frame(frame_controles, bg='white')
        frame_dim.pack(fill='x', padx=20, pady=10)
        
        tk.Label(frame_dim, text="Ancho:", bg='white').pack(side='left')
        self.ancho_var = tk.StringVar(value="256")
        tk.Entry(frame_dim, textvariable=self.ancho_var, width=8).pack(side='left', padx=5)
        
        tk.Label(frame_dim, text="Alto:", bg='white').pack(side='left', padx=(20,0))
        self.alto_var = tk.StringVar(value="256")
        tk.Entry(frame_dim, textvariable=self.alto_var, width=8).pack(side='left', padx=5)
        
        # Checkboxes
        self.incluir_cortical = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_controles, text="Incluir hueso cortical", 
                      variable=self.incluir_cortical, bg='white').pack(anchor='w', padx=20, pady=5)
        
        # Semilla
        frame_semilla = tk.Frame(frame_controles, bg='white')
        frame_semilla.pack(fill='x', padx=20, pady=5)
        tk.Label(frame_semilla, text="Semilla (opcional):", bg='white').pack(side='left')
        self.semilla_var = tk.StringVar(value="42")
        tk.Entry(frame_semilla, textvariable=self.semilla_var, width=10).pack(side='left', padx=5)
        
        # Botones
        frame_botones = tk.Frame(frame_controles, bg='white')
        frame_botones.pack(fill='x', padx=20, pady=15)
        
        tk.Button(frame_botones, text="Generar Imagen Individual", 
                 command=self.generar_imagen_individual, bg='#4CAF50', fg='white', 
                 font=("Arial", 10, "bold")).pack(side='left', padx=5)
        
        tk.Button(frame_botones, text="Generar Dataset", 
                 command=self.generar_dataset, bg='#2196F3', fg='white', 
                 font=("Arial", 10, "bold")).pack(side='left', padx=5)
        
        tk.Button(frame_botones, text="Comparar Tipos", 
                 command=self.comparar_tipos, bg='#FF9800', fg='white', 
                 font=("Arial", 10, "bold")).pack(side='left', padx=5)
        
        # Área de visualización
        self.frame_visualizacion = tk.Frame(frame_sim, bg='white', relief='sunken', bd=2)
        self.frame_visualizacion.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.label_imagen_sim = tk.Label(self.frame_visualizacion, text="Las imágenes generadas aparecerán aquí", 
                                        bg='white', fg='#666')
        self.label_imagen_sim.pack(expand=True)
    
    def crear_pestana_analisis_ks(self):
        """Crea la pestaña de análisis KS."""
        frame_ks = ttk.Frame(self.notebook)
        self.notebook.add(frame_ks, text="Análisis Kolmogorov-Smirnov")
        
        # Frame superior para selección de imágenes
        frame_superior = tk.Frame(frame_ks, bg='white', relief='raised', bd=2)
        frame_superior.pack(fill='x', padx=10, pady=10)
        
        tk.Label(frame_superior, text="Selección de Imágenes para Análisis", 
                font=("Arial", 14, "bold"), bg='white').pack(pady=10)
        
        # Opción: Simulaciones vs Archivos
        self.modo_imagenes = tk.StringVar(value="simulaciones")
        frame_modo = tk.Frame(frame_superior, bg='white')
        frame_modo.pack(fill='x', padx=20, pady=10)
        
        tk.Radiobutton(frame_modo, text="Usar Simulaciones", variable=self.modo_imagenes, 
                      value="simulaciones", command=self.cambiar_modo_imagenes, bg='white').pack(side='left', padx=20)
        tk.Radiobutton(frame_modo, text="Cargar Imágenes de Archivo", variable=self.modo_imagenes, 
                      value="archivos", command=self.cambiar_modo_imagenes, bg='white').pack(side='left', padx=20)
        
        # Frame para opciones específicas
        self.frame_opciones_img = tk.Frame(frame_superior, bg='white')
        self.frame_opciones_img.pack(fill='x', padx=20, pady=10)
        
        # Inicializar con modo simulaciones
        self.crear_opciones_simulaciones()
        
        # Botón para realizar análisis
        tk.Button(frame_superior, text="Realizar Análisis KS", 
                 command=self.realizar_analisis_ks, bg='#E91E63', fg='white', 
                 font=("Arial", 12, "bold")).pack(pady=15)
        
        # Frame para mostrar imágenes y resultados
        frame_resultados = tk.Frame(frame_ks, bg='white', relief='sunken', bd=2)
        frame_resultados.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Dividir en dos columnas: imágenes y resultados
        self.frame_imagenes_ks = tk.Frame(frame_resultados, bg='white')
        self.frame_imagenes_ks.pack(side='left', fill='both', expand=True)
        
        self.frame_resultados_ks = tk.Frame(frame_resultados, bg='#f8f8f8', width=300)
        self.frame_resultados_ks.pack(side='right', fill='y', padx=10, pady=10)
        
        # Labels para imágenes
        tk.Label(self.frame_imagenes_ks, text="Imágenes a Comparar", 
                font=("Arial", 12, "bold"), bg='white').pack(pady=10)
        
        self.label_resultado_ks = tk.Label(self.frame_resultados_ks, 
                                          text="Resultados del análisis aparecerán aquí", 
                                          bg='#f8f8f8', justify='left', wraplength=280)
        self.label_resultado_ks.pack(anchor='nw', padx=10, pady=10)
    
    def cambiar_modo_imagenes(self):
        """Cambia entre modo simulaciones y archivos."""
        # Limpiar frame de opciones
        for widget in self.frame_opciones_img.winfo_children():
            widget.destroy()
        
        if self.modo_imagenes.get() == "simulaciones":
            self.crear_opciones_simulaciones()
        else:
            self.crear_opciones_archivos()
    
    def crear_opciones_simulaciones(self):
        """Crea las opciones para usar simulaciones."""
        tk.Label(self.frame_opciones_img, text="Selecciona los tipos de tejido a comparar:", 
                bg='white').pack(anchor='w', pady=5)
        
        frame_tipos = tk.Frame(self.frame_opciones_img, bg='white')
        frame_tipos.pack(fill='x', pady=5)
        
        # Primera imagen
        frame_img1 = tk.Frame(frame_tipos, bg='white')
        frame_img1.pack(side='left', fill='x', expand=True)
        tk.Label(frame_img1, text="Imagen 1:", bg='white', font=("Arial", 10, "bold")).pack(anchor='w')
        self.tipo1_var = tk.StringVar(value="normal")
        tk.Radiobutton(frame_img1, text="Normal", variable=self.tipo1_var, value="normal", bg='white').pack(anchor='w')
        tk.Radiobutton(frame_img1, text="Osteopenia", variable=self.tipo1_var, value="osteopenia", bg='white').pack(anchor='w')
        tk.Radiobutton(frame_img1, text="Osteoporosis", variable=self.tipo1_var, value="osteoporosis", bg='white').pack(anchor='w')
        
        # Segunda imagen
        frame_img2 = tk.Frame(frame_tipos, bg='white')
        frame_img2.pack(side='right', fill='x', expand=True)
        tk.Label(frame_img2, text="Imagen 2:", bg='white', font=("Arial", 10, "bold")).pack(anchor='w')
        self.tipo2_var = tk.StringVar(value="osteoporosis")
        tk.Radiobutton(frame_img2, text="Normal", variable=self.tipo2_var, value="normal", bg='white').pack(anchor='w')
        tk.Radiobutton(frame_img2, text="Osteopenia", variable=self.tipo2_var, value="osteopenia", bg='white').pack(anchor='w')
        tk.Radiobutton(frame_img2, text="Osteoporosis", variable=self.tipo2_var, value="osteoporosis", bg='white').pack(anchor='w')
    
    def crear_opciones_archivos(self):
        """Crea las opciones para cargar archivos."""
        # Botones para cargar imágenes
        frame_carga = tk.Frame(self.frame_opciones_img, bg='white')
        frame_carga.pack(fill='x', pady=10)
        
        tk.Button(frame_carga, text="Cargar Imagen 1", 
                 command=lambda: self.cargar_imagen(1), bg='#4CAF50', fg='white').pack(side='left', padx=10)
        tk.Button(frame_carga, text="Cargar Imagen 2", 
                 command=lambda: self.cargar_imagen(2), bg='#4CAF50', fg='white').pack(side='left', padx=10)
        
        # Labels para mostrar archivos cargados
        self.label_archivo1 = tk.Label(self.frame_opciones_img, text="Imagen 1: No cargada", 
                                      bg='white', fg='#666')
        self.label_archivo1.pack(anchor='w', padx=10, pady=2)
        
        self.label_archivo2 = tk.Label(self.frame_opciones_img, text="Imagen 2: No cargada", 
                                      bg='white', fg='#666')
        self.label_archivo2.pack(anchor='w', padx=10, pady=2)
    
    def cargar_imagen(self, numero):
        """Carga una imagen desde archivo."""
        archivo = filedialog.askopenfilename(
            title=f"Seleccionar Imagen {numero}",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Todos", "*.*")]
        )
        
        if archivo:
            try:
                imagen = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
                if imagen is None:
                    raise ValueError("No se pudo cargar la imagen")
                
                if numero == 1:
                    self.imagen1 = imagen
                    self.label_archivo1.config(text=f"Imagen 1: {archivo.split('/')[-1]}", fg='#4CAF50')
                else:
                    self.imagen2 = imagen
                    self.label_archivo2.config(text=f"Imagen 2: {archivo.split('/')[-1]}", fg='#4CAF50')
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")
    
    def generar_imagen_individual(self):
        """Genera una imagen individual según parámetros."""
        try:
            ancho = int(self.ancho_var.get())
            alto = int(self.alto_var.get())
            semilla = int(self.semilla_var.get()) if self.semilla_var.get() else None
            
            # Convertir string a TipoTejido
            tipo = gt.TipoTejido(self.tipo_tejido.get())
            
            # Generar imagen usando el generador de tejidos
            imagen = gt.generar_imagen_hueso_simulada(
                width=ancho,
                height=alto,
                tipo=tipo,
                incluir_cortical=self.incluir_cortical.get(),
                seed=semilla
            )
            
            self.mostrar_imagen_simulada(imagen, f"Tejido {tipo.value.capitalize()}")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Error en los parámetros: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar imagen: {str(e)}")

    def generar_dataset(self):
        """Genera un dataset completo."""
        try:
            # Obtener parámetros
            ancho = int(self.ancho_var.get())
            alto = int(self.alto_var.get())
            semilla = int(self.semilla_var.get()) if self.semilla_var.get() else None
            
            # Crear directorio para el dataset si no existe
            import os
            directorio = filedialog.askdirectory(title="Seleccionar directorio para dataset")
            if not directorio:
                return
                
            # Generar dataset
            dataset = gt.generar_dataset_simulado(
                n_imagenes_por_tipo=5,  # Número de imágenes por tipo
                dimensiones=(ancho, alto),
                incluir_cortical=self.incluir_cortical.get(),
                seed_base=semilla
            )
            
            # Guardar imágenes
            for tipo, imagenes in dataset.items():
                tipo_dir = os.path.join(directorio, tipo)
                os.makedirs(tipo_dir, exist_ok=True)
                
                for i, imagen in enumerate(imagenes):
                    ruta = os.path.join(tipo_dir, f"{tipo}_{i+1}.png")
                    cv2.imwrite(ruta, imagen)
            
            messagebox.showinfo("Éxito", f"Dataset generado en:\n{directorio}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar dataset: {str(e)}")
    
    def comparar_tipos(self):
        """Genera y muestra comparación de los tres tipos."""
        try:
            # Limpiar el frame
            for widget in self.frame_visualizacion.winfo_children():
                widget.destroy()
            
            tk.Label(self.frame_visualizacion, text="Comparación de Tipos de Tejido", 
                    font=("Arial", 14, "bold"), bg='white').pack(pady=10)
            
            frame_imagenes = tk.Frame(self.frame_visualizacion, bg='white')
            frame_imagenes.pack()

            ancho = int(self.ancho_var.get())
            alto = int(self.alto_var.get())
            semilla = int(self.semilla_var.get()) if self.semilla_var.get() else None
            
            for i, tipo in enumerate(gt.TipoTejido):
                frame_tipo = tk.Frame(frame_imagenes, bg='white')
                frame_tipo.pack(side='left', padx=10)
                
                # Generar imagen usando el generador
                imagen = gt.generar_imagen_hueso_simulada(
                    width=ancho,
                    height=alto,
                    tipo=tipo,
                    incluir_cortical=self.incluir_cortical.get(),
                    seed=semilla + i
                )
                
                imagen_pil = Image.fromarray(imagen)
                imagen_pil = imagen_pil.resize((200, 200), Image.Resampling.LANCZOS)
                imagen_tk = ImageTk.PhotoImage(imagen_pil)
                
                tk.Label(frame_tipo, text=tipo.value.capitalize(), 
                        font=("Arial", 10, "bold"), bg='white').pack()
                label_img = tk.Label(frame_tipo, image=imagen_tk, bg='white')
                label_img.image = imagen_tk
                label_img.pack()
                
                # Mostrar estadísticas básicas
                stats_text = f"Media: {imagen.mean():.1f}\nStd: {imagen.std():.1f}"
                tk.Label(frame_tipo, text=stats_text, 
                        font=("Arial", 8), bg='white').pack()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al comparar tipos: {str(e)}")

    def realizar_analisis_ks(self):
        """Realiza el análisis KS según el modo seleccionado."""
        try:
            if self.modo_imagenes.get() == "simulaciones":
                self.generar_imagenes_para_ks()
            
            if self.imagen1 is None or self.imagen2 is None:
                messagebox.showerror("Error", "Faltan imágenes para el análisis")
                return
            
            # Realizar prueba KS
            self.resultados_ks = ks.prueba_ks_imagenes(self.imagen1, self.imagen2)
            
            # Mostrar resultados
            self.mostrar_resultados_ks(self.resultados_ks)
            self.mostrar_imagenes_ks()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en análisis KS: {str(e)}")
    
    def generar_imagenes_para_ks(self):
        """Genera las imágenes simuladas para el análisis KS."""
        try:

            ancho = int(self.ancho_var.get())
            alto = int(self.alto_var.get())

            # Convertir strings a TipoTejido
            tipo1 = gt.TipoTejido(self.tipo1_var.get())
            tipo2 = gt.TipoTejido(self.tipo2_var.get())
            
            # Generar imágenes
            semilla = int(self.semilla_var.get()) if self.semilla_var.get() else None
            
            self.imagen1 = gt.generar_imagen_hueso_simulada(
                width=ancho, height=alto,
                tipo=tipo1,
                incluir_cortical=self.incluir_cortical.get(),
                seed=semilla
            )
            
            self.imagen2 = gt.generar_imagen_hueso_simulada(
                width=ancho, height=alto,
                tipo=tipo2,
                incluir_cortical=self.incluir_cortical.get(),
                seed=semilla + 1  # Asegurar que sea diferente
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar imágenes: {str(e)}")
            raise e
    
    def mostrar_imagen_simulada(self, imagen, titulo):
        """Muestra una imagen simulada en el área de visualización."""
        # Limpiar el frame
        for widget in self.frame_visualizacion.winfo_children():
            widget.destroy()
        
        # Convertir imagen para Tkinter
        imagen_pil = Image.fromarray(imagen)
        imagen_pil = imagen_pil.resize((300, 300), Image.Resampling.LANCZOS)
        imagen_tk = ImageTk.PhotoImage(imagen_pil)
        
        # Mostrar imagen
        tk.Label(self.frame_visualizacion, text=titulo, font=("Arial", 12, "bold"), 
                bg='white').pack(pady=10)
        label_img = tk.Label(self.frame_visualizacion, image=imagen_tk, bg='white')
        label_img.image = imagen_tk  # Mantener referencia
        label_img.pack()
    
    def mostrar_comparacion_tipos(self):
        """Muestra comparación de los tres tipos de tejido."""
        # Limpiar el frame
        for widget in self.frame_visualizacion.winfo_children():
            widget.destroy()
        
        tk.Label(self.frame_visualizacion, text="Comparación de Tipos de Tejido", 
                font=("Arial", 14, "bold"), bg='white').pack(pady=10)
        
        frame_imagenes = tk.Frame(self.frame_visualizacion, bg='white')
        frame_imagenes.pack()
        
        tipos = ["Normal", "Osteopenia", "Osteoporosis"]
        for i, tipo in enumerate(tipos):
            frame_tipo = tk.Frame(frame_imagenes, bg='white')
            frame_tipo.pack(side='left', padx=10)
            
            # Simular imagen
            np.random.seed(40 + i)
            imagen = np.random.randint(50 + i*30, 200 - i*30, (128, 128), dtype=np.uint8)
            
            imagen_pil = Image.fromarray(imagen)
            imagen_pil = imagen_pil.resize((150, 150), Image.Resampling.LANCZOS)
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
            
            tk.Label(frame_tipo, text=tipo, font=("Arial", 10, "bold"), bg='white').pack()
            label_img = tk.Label(frame_tipo, image=imagen_tk, bg='white')
            label_img.image = imagen_tk
            label_img.pack()
    
    def mostrar_imagenes_ks(self):
        """Muestra las imágenes que se están comparando y sus gráficos."""
        # Limpiar frame de imágenes
        for widget in self.frame_imagenes_ks.winfo_children():
            widget.destroy()
        
        # Frame para imágenes
        frame_imgs = tk.Frame(self.frame_imagenes_ks, bg='white')
        frame_imgs.pack(fill='x', pady=10)
        
        # Mostrar ambas imágenes
        for i, (imagen, titulo) in enumerate([(self.imagen1, "Imagen 1"), (self.imagen2, "Imagen 2")]):
            frame_img = tk.Frame(frame_imgs, bg='white')
            frame_img.pack(side='left', padx=20)
            
            imagen_pil = Image.fromarray(imagen)
            imagen_pil = imagen_pil.resize((200, 200), Image.Resampling.LANCZOS)
            imagen_tk = ImageTk.PhotoImage(imagen_pil)
            
            tk.Label(frame_img, text=titulo, font=("Arial", 10, "bold"), bg='white').pack()
            label_img = tk.Label(frame_img, image=imagen_tk, bg='white')
            label_img.image = imagen_tk
            label_img.pack()
        
        # Frame para gráficos
        frame_graficos = tk.Frame(self.frame_imagenes_ks, bg='white')
        frame_graficos.pack(fill='both', expand=True, pady=10)
        
        # Crear figura con dos subplots
        fig = Figure(figsize=(10, 4))
        
        # Histogramas
        ax1 = fig.add_subplot(121)
        hist1 = ks.calcular_histograma_normalizado(self.imagen1)
        hist2 = ks.calcular_histograma_normalizado(self.imagen2)
        x = np.arange(256)
        
        ax1.plot(x, hist1, label='Imagen 1', color='blue')
        ax1.plot(x, hist2, label='Imagen 2', color='red', linestyle='--')
        ax1.set_title('Histogramas')
        ax1.set_xlabel('Nivel de gris')
        ax1.set_ylabel('Frecuencia relativa')
        ax1.legend()
        
        # FDA
        ax2 = fig.add_subplot(122)
        fda1 = ks.calcular_fda_empirica(hist1)
        fda2 = ks.calcular_fda_empirica(hist2)
        
        ax2.plot(x, fda1, label='Imagen 1', color='blue')
        ax2.plot(x, fda2, label='Imagen 2', color='red', linestyle='--')
        ax2.set_title('Función de Distribución Acumulada')
        ax2.set_xlabel('Nivel de gris')
        ax2.set_ylabel('Probabilidad acumulada')
        ax2.legend()
        
        # Mostrar D máximo
        if hasattr(self, 'resultados_ks'):
            d_max = self.resultados_ks['estadistico_D']
            idx_max = np.argmax(np.abs(fda1 - fda2))
            ax2.plot([x[idx_max], x[idx_max]], 
                    [min(fda1[idx_max], fda2[idx_max]), 
                     max(fda1[idx_max], fda2[idx_max])],
                    'k--', label=f'D={d_max:.3f}')
            ax2.legend()
        
        fig.tight_layout()
        
        # Crear canvas y mostrarlo
        canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def mostrar_resultados_ks(self, resultados):
        """Muestra los resultados del análisis KS."""
        # Limpiar resultados anteriores
        for widget in self.frame_resultados_ks.winfo_children():
            widget.destroy()
        
        # Título
        tk.Label(self.frame_resultados_ks, text="Resultados del Test KS", 
                font=("Arial", 12, "bold"), bg='#f8f8f8').pack(pady=10)
        
        # Estadísticas principales
        stats_text = f"""
Estadístico D:  {resultados['estadistico_D']:.4f}
Lambda KS:      {resultados['lambda_KS']:.4f}
P-valor:        {resultados['p_valor']:.4f}
Alpha:          {resultados['alpha']:.4f}

Decisión:
{"Se rechaza H₀" if resultados['rechazar_H0'] else "No se rechaza H₀"}
"""
        
        tk.Label(self.frame_resultados_ks, text=stats_text, 
                font=("Courier", 10), bg='#f8f8f8', justify='left').pack(padx=10)
        
        # Conclusión
        conclusion_frame = tk.Frame(self.frame_resultados_ks, bg='#f8f8f8')
        conclusion_frame.pack(fill='x', padx=10, pady=10)
        
        color = '#d32f2f' if resultados['rechazar_H0'] else '#388e3c'
        tk.Label(conclusion_frame, text="Conclusión:", 
                font=("Arial", 10, "bold"), bg='#f8f8f8').pack(anchor='w')
        tk.Label(conclusion_frame, text=resultados['conclusion'],
                font=("Arial", 9), bg='#f8f8f8', fg=color,
                wraplength=250, justify='left').pack(pady=(5,0))

def main():
    """Función principal para ejecutar la aplicación."""
    root = tk.Tk()
    app = DetectorOsteoporosisApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
