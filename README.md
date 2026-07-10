# Gemelo Digital para Optimización de Control en Beamline de Iones de Silicio
### Proyecto - Hackathon MIT & Universidad Nacional de Colombia (UNAL)

Este repositorio contiene la implementación de un **Gemelo Digital (Digital Twin)** para emular físicamente y controlar en lazo cerrado una línea de transporte de haz de iones de silicio (beamline) utilizando el simulador físico **SIMION** y técnicas avanzadas de Deep Learning basadas en el paper *CBOL-Tuner*.

---

## 🚀 Características Principales
* **Reducción de Tiempo de Simulación**: Mapea la física compleja de trayectorias de iones que tarda ~40 segundos en SIMION a inferencias en milisegundos ($<1$ ms) en PyTorch.
* **Espacio Latente de Dimensiones Reducidas**: Compresión del perfil 2D del haz ($X \in [0, 1]^{400}$) a una variedad de baja dimensión ($z \in \mathbb{R}^2$) utilizando un Autoencoder Variacional ($\beta$-VAE).
* **Filtro de Densidad KDE**: Estimación de log-verosimilitud mediante *Kernel Density Estimation* para evitar la inestabilidad física o configuraciones no válidas (*Out-of-Distribution*) en la decodificación.
* **Control Inverso de Setpoint**: Deducción automática de los voltajes de los 8 electrodos libres a partir de la forma de haz deseada usando retropropagación a través del emulador Forward.
* **Optimización por Espacio Latente (C-BO)**: Optimización bayesiana Optuna en la variedad física latente filtrada con KDE y refinada con descenso de gradiente.

---

## 📂 Estructura del Proyecto

```text
Digital Twin/
│
├── README.md                          # Este archivo explicativo.
├── use_digital_twin.py                # Script CLI/API interactivo para predicción, inversión y optimización.
├── digital_twin_flow.html             # Dashboard web interactivo para demostración visual de las 3 fases.
│
├── Hackathon_student/
│   └── beamline_dataset.npz           # Conjunto de datos consolidado (486 muestras reales LHS + Perturbaciones).
│
├── Figuras/                           # Gráficas de diagnóstico de entrenamiento y validación.
│   ├── vae_reconstruction.png         # Reconstrucción espacial VAE (Real vs Reconstruido).
│   ├── latent_space_scatter.png       # Variedad y clusters del espacio latente.
│   ├── phase_transition_grokking.png  # Curva de Grokking en el VAE con la rodilla Nc.
│   └── dnn_vs_vae_latent.png          # Correlación de coordenadas z (Reales vs Predichas).
│
├── Informe/                           # Documentación técnica y reportes (compilables en pdfLaTeX).
│   ├── Datos/                         # Informe detallado sobre el proceso de recolección de datos LHS.
│   ├── Fase_1/                        # Informe detallado sobre el diseño y entrenamiento del VAE (Fase 1).
│   ├── Fase_2/                        # Informe detallado sobre el diseño y entrenamiento del Emulador Forward (Fase 2).
│   ├── Fase_3_control/                # Informe detallado sobre el Estimador Inverso y la optimización C-BO (Fase 3).
│   ├── Grok_Densificación/            # Informe detallado sobre el fenómeno de Grokking y Densificación Local.
│   └── Pipeline_completo/
│       ├── informe_final.tex          # Documento LaTeX del reporte técnico unificado de 14 páginas.
│       └── informe_final.pdf          # Reporte técnico compilado final con figuras y tablas vectoriales.
│
└── src/                               # Código fuente del pipeline de entrenamiento y control.
    ├── dataset.py                     # Carga del dataset y normalización de voltajes/histogramas.
    ├── retrain_all.py                 # Pipeline integrador automatizado de 5 pasos para re-entrenar modelos.
    ├── generate_plots.py              # Script para regenerar todas las gráficas diagnósticas.
    │
    ├── phase_1_VAE/                   # Fase 1: Compresión del espacio del haz.
    │   ├── vae.py                     # Arquitectura de la red VAE.
    │   ├── collect_all_real_data.py   # Script acoplado a SIMION para recolectar datos reales.
    │   └── train_vae.py               # Rutina de optimización de pesos del VAE.
    │
    ├── phase_2_DNN/                   # Fase 2: Emulador del comportamiento físico (y -> z).
    │   ├── model.py                   # Red ForwardRegressor y FullEmulator.
    │   └── train_dnn.py               # Entrenamiento del emulador forward.
    │
    └── phase_3_control/               # Fase 3: Optimización latente e inverso.
        ├── inverse_estimator.py       # Red del estimador inverso de voltajes (z -> y).
        └── control_optimizer.py       # Algoritmo de optimización C-BO con Optuna, KDE y retropropagación.
```

---

## 🛠️ Requisitos de Instalación

1. **Instalación de Librerías de Python**:
   Se recomienda crear un entorno virtual e instalar las dependencias:
   ```bash
   pip install torch numpy matplotlib scikit-learn optuna scipy
   ```

2. **SIMION (Opcional - Requerido solo para recolectar datos o validar)**:
   Asegurar que `simion.exe` esté disponible en las variables de entorno (`PATH`) y que los archivos geométricos y cinemáticos (`.pa0`, `.iob`) estén en la carpeta de trabajo correspondiente.

---

## 🎮 Guía de Uso del Script Interactivo (`use_digital_twin.py`)

Se ha diseñado una interfaz de consola sumamente simple para que cualquier persona o jurado evalúe el Gemelo Digital en milisegundos en **CPU** sin necesidad de SIMION.

### 1. Predicción Forward (Emulador)
Calcula el perfil de haz espacial predicho a partir de los 8 voltajes en el detector.
```bash
python use_digital_twin.py --mode predict --voltages -576 -329 281 0 104 -563 742 -10
```
* **Salida**: Genera la predicción geométrica e imprime la transmisión esperada. Guarda el perfil en **`predicted_profile.png`**.

### 2. Control Inverso (Consigna Inversa)
Calcula los voltajes ideales que producen una forma del haz objetivo de entrada (por ejemplo, el caso del Trial 104 del dataset).
```bash
python use_digital_twin.py --mode invert --trial 104
```
* **Salida**: Imprime la tabla comparativa de voltajes (Originales vs. Inferidos). Exporta **`target_profile.png`** y **`reconstructed_profile.png`** para comparar el haz geométricamente.

### 3. Optimización de Dirección (Steering)
Busca la configuración óptima para maximizar los impactos en la variedad física.
```bash
python use_digital_twin.py --mode optimize
```
* **Salida**: Imprime los 8 voltajes ideales calculados y exporta la forma óptima predicha a **`optimized_profile.png`**.

*(Añade el flag `--simion` al final de cualquier comando si tienes SIMION configurado y quieres validar la física real).*

---

## 🖥️ Dashboard Interactivo Web (`digital_twin_flow.html`)

Para una sustentación interactiva frente a los jurados, abre el archivo **`digital_twin_flow.html`** haciendo doble clic sobre él en tu computadora. Este cuenta con:
* **Simulador de Haz con Sliders**: Modifica los 8 voltajes y observa cómo el haz de iones se desplaza, se desenfoca o colima en una pantalla de calor de magma.
* **KDE Latent Clicker**: Haz clic en cualquier parte del espacio latente y el sistema calculará al instante los voltajes inversos correspondientes (moviendo los deslizadores automáticamente) y te alertará mediante un indicador luminoso si el estado es físicamente estable según el KDE.
* **Diagrama SVG animado**: Haz clic en cualquier bloque de la arquitectura para entender la teoría y el código que actúa por detrás.

---

## 📊 Resumen de Resultados del Entrenamiento
* **Fase 1 (VAE)**: MSE de reconstrucción bajo, con un ELBO final de **$0.057009$**. El espacio latente separa limpiamente los impactos altos de los nulos.
* **Fase 2 (Forward)**: MSE en validación de coordenadas latentes de **$0.256195$**.
* **Fase 3 (Inverse)**: MSE en validación de voltajes de **$0.091156$** (entrenado únicamente con muestras útiles).
* **Validación de Control Inverso (Reconstrucción del Trial 104)**: El Gemelo Digital infirió voltajes que, al validarse directamente en SIMION, lograron **130 de 500 impactos reales (26.0% de transmisión)** con un **Spread altamente colimado de $2.749$**, frente a los 174 impactos originales.
