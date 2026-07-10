# Gemelo Digital para Optimización de Control en Beamline de Iones de Silicio
### Proyecto - Hackathon MIT & Universidad Nacional de Colombia (UNAL)

Este repositorio contiene la implementación de un **Gemelo Digital (Digital Twin)** para emular físicamente y controlar en lazo cerrado una línea de transporte de haz de iones de silicio (beamline) utilizando el simulador físico **SIMION** y técnicas avanzadas de Deep Learning basadas en el paper *CBOL-Tuner*.

---

## 🚀 Características Principales
* **Reducción de Tiempo de Simulación**: Mapea la física compleja de trayectorias de iones que tarda ~40 segundos en SIMION a inferencias en milisegundos ($<1$ ms) en PyTorch.
* **Espacio Latente de Dimensiones Reducidas**: Compresión del perfil 2D del haz ($X \in [0, 1]^{400}$) a una variedad continua en 4D ($z \in \mathbb{R}^4$) utilizando un Autoencoder Variacional ($\beta$-VAE) con salida Softmax y pérdida de Entropía Cruzada Multiclase.
* **Emulación Forward Desacoplada**: Predicción en paralelo de la topología latente del haz ($y \to z \in \mathbb{R}^4$) y de la transmisión escalar ($y \to T \in [0, 1]$) con un emulador compuesto diferenciable de extremo a extremo.
* **Filtro de Densidad KDE**: Estimación de log-verosimilitud mediante *Kernel Density Estimation* en 4D para evitar la inestabilidad física o configuraciones no válidas (*Out-of-Distribution*) en la decodificación.
* **Control Inverso de Setpoint**: Deducción automática de los voltajes de los 8 electrodos libres a partir de la forma de haz deseada usando un estimador inverso z-to-y y retropropagación a través del emulador Forward.
* **Optimización por Espacio Latente (C-BO)**: Optimización bayesiana Optuna en la variedad física latente filtrada con KDE, acelerada por semillas (*warm-starting*) de ensayos históricos, y refinada localmente con descenso de gradiente.

---

## 📂 Estructura del Proyecto

```text
Digital Twin/
│
├── README.md                          # Este archivo explicativo.
├── use_digital_twin.py                # Script CLI/API interactivo para predicción, inversión y optimización.
├── digital_twin_flow.html             # Dashboard web interactivo para demostración visual de las 3 fases.
├── informe_hackathon.tex              # Archivo fuente de LaTeX del reporte ejecutivo.
├── informe_hackathon.pdf              # Reporte técnico ejecutivo compilado final (17 páginas).
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
python use_digital_twin.py --mode predict --voltages -697.29 -656.61 -169.21 280.02 260.67 -333.86 391.68 -671.93
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

## 📊 Resumen de Resultados del Entrenamiento
* **Fase 1 (VAE)**: Pérdida de validación total ELBO (Entropía Cruzada + KL) de **$5.524675$** sobre el espacio latente 4D.
* **Fase 2 (Forward)**: MSE en validación latente de **$1.249795$** para las coordenadas latentes 4D y MSE escalar de transmisión de **$0.000213$**.
* **Fase 3 (Inverse)**: MSE de validación de **$0.034058$** sobre el estimador inverso en 4D (mejora de precisión del 400% respecto a la versión 2D).
* **Validación de Control Inverso (Consigna Inversa - Trial 104)**: El Gemelo Digital infirió voltajes que, al validarse físicamente en SIMION, lograron **229 de 500 impactos reales ($45.8\%$ de transmisión)** con un **Spread altamente colimado de $2.774$**, superando en 55 impactos (+31.6% de eficiencia de corriente) la transmisión del haz original.
* **Validación de Steering (Dirección del Haz)**: La optimización C-BO y el refinamiento por gradiente lograron **215 de 500 impactos reales ($43.0\%$ de transmisión)** y un spread de $2.960$.
