import os
import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube

# Configuración de rutas para importación
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(CURRENT_DIR))
sys.path.append(str(WORKSPACE_ROOT / "Hackathon_student"))

# Intentar importar el VAE real y utilidades del simulador
try:
    from vae import VAE
    from dataset import run_single_simulation, build_histogram, OPTIMIZE, FIXED
    REAL_IMPORTS_AVAILABLE = True
except ImportError:
    REAL_IMPORTS_AVAILABLE = False


# =============================================================================
#  MÓDULO 1: Generación de Datos con Latin Hypercube Sampling (LHS)
# =============================================================================
def generar_muestras_lhs(n_samples, bounds_min, bounds_max, seed=None):
    """
    Genera n_samples de combinaciones de voltajes usando Latin Hypercube Sampling (LHS)
    para garantizar una cobertura ortogonal en un hipercubo de 8 dimensiones.
    
    Parámetros:
      n_samples (int): Número de configuraciones a generar.
      bounds_min (list/array): Límites mínimos para cada dimensión (longitud 8).
      bounds_max (list/array): Límites máximos para cada dimensión (longitud 8).
      seed (int, opcional): Semilla aleatoria para la reproducibilidad.
      
    Retorna:
      pd.DataFrame: DataFrame con las configuraciones de voltajes (V3, V6, V9, V10, V11, V12, V15, V18).
    """
    d = len(bounds_min)
    assert d == 8, "El espacio de voltajes debe tener exactamente 8 dimensiones."
    
    # Inicializar el sampler de LHS
    sampler = LatinHypercube(d=d, seed=seed)
    samples = sampler.random(n=n_samples)
    
    # Escalar las muestras del rango [0, 1] al rango físico [bounds_min, bounds_max]
    bounds_min = np.array(bounds_min)
    bounds_max = np.array(bounds_max)
    scaled_samples = bounds_min + samples * (bounds_max - bounds_min)
    
    # Electrodos libres de interés (ordenados de forma consistente con el proyecto)
    opt_keys = [3, 6, 9, 10, 11, 12, 15, 18]
    cols = [f"V{k}" for k in opt_keys]
    
    return pd.DataFrame(scaled_samples, columns=cols)


# =============================================================================
#  SIMULADOR / EVALUADOR DE COMPORTAMIENTO (Física y Modo Sintético/Real)
# =============================================================================
def evaluar_simion(voltajes, use_real_simion=False):
    """
    Evalúa la configuración de voltajes especificada.
    Si use_real_simion es True, ejecuta la simulación real de SIMION en la máquina.
    De lo contrario, evalúa un modelo sintético continuo que emula la física del haz.
    
    Parámetros:
      voltajes (array-like o dict): Configuración de voltajes para los 8 electrodos.
      use_real_simion (bool): Si es True, corre el simulador SIMION.
      
    Retorna:
      np.ndarray: Histograma 2D aplanado (400 dimensiones).
      float: Fracción de transmisión (0.0 a 1.0).
    """
    opt_keys = [3, 6, 9, 10, 11, 12, 15, 18]
    
    # Conversión de formato de entrada a diccionario y array
    if isinstance(voltajes, dict):
        voltages_dict = voltajes
        voltages_array = np.array([voltajes.get(k, 0.0) for k in opt_keys])
    else:
        voltages_array = np.array(voltajes)
        voltages_dict = {opt_keys[i]: float(voltages_array[i]) for i in range(8)}
        
    # --- MODO REAL SIMION ---
    if use_real_simion and REAL_IMPORTS_AVAILABLE:
        try:
            positions = run_single_simulation(voltages_dict)
            histogram = build_histogram(positions)
            transmission = float(histogram.sum())  # Suma de la transmisión normalizada
            return histogram, transmission
        except Exception as e:
            print(f"[Warning] Error ejecutando SIMION real: {e}. Usando fallback sintético.")
            
    # --- MODO SINTÉTICO (Emulador físico continuo para demostración rápida) ---
    # Voltaje objetivo conocido como un punto óptimo de transmisión (ej. de STARTING_POINT)
    optimo = np.array([-671.4, -173.0, 874.4, -410.0, 124.7, -843.5, 566.7, 663.3])
    
    # Calculamos la distancia al óptimo en un espacio normalizado
    dist = np.linalg.norm(voltages_array - optimo) / 1000.0
    transmission = np.exp(-dist * dist * 3.0)  # Curva de campana para simular transmisión física
    
    # Generar un histograma 2D 20x20
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    
    # El haz se desvía levemente según los voltajes aplicados
    offset_x = 0.4 * (voltages_array[0] + voltages_array[1]) / 1000.0
    offset_y = 0.4 * (voltages_array[2] + voltages_array[3]) / 1000.0
    
    # Simulación del haz como un haz gaussiano enfocado en el detector
    blob = np.exp(-((X - offset_x)**2 + (Y - offset_y)**2) / 0.4)
    if blob.sum() > 0:
        blob = blob / blob.sum()
        
    histogram = (blob * transmission).flatten().astype(np.float32)
    
    # Añadir un ruido Gaussiano muy sutil para simular imperfecciones de medición
    ruido = np.random.normal(0, 0.005, size=histogram.shape).astype(np.float32)
    histogram = np.clip(histogram + ruido, 0.0, 1.0)
    
    return histogram, transmission


# =============================================================================
#  MÓDULO 2: Bucle de Detección del "Cambio de Fase" (Phase Transition)
# =============================================================================
class LocalDataset(torch.utils.data.Dataset):
    """Dataset auxiliar de PyTorch para entrenar el VAE."""
    def __init__(self, histograms):
        self.histograms = torch.tensor(histograms, dtype=torch.float32)
        
    def __len__(self):
        return len(self.histograms)
        
    def __getitem__(self, idx):
        # Retorna un vector dummy de voltajes (8) y el histograma (400)
        # Esto mantiene la firma (voltajes, histogramas) idéntica a BeamlineDataset
        dummy_voltage = torch.zeros(8, dtype=torch.float32)
        return dummy_voltage, self.histograms[idx]


def iniciar_modelo_vae(latent_dim=2):
    """
    Crea e inicializa una instancia del VAE.
    Intenta importar la clase real VAE. Si no está disponible, inicializa una versión local.
    """
    if REAL_IMPORTS_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return VAE(input_dim=400, latent_dim=latent_dim).to(device)
    else:
        # Fallback local de la arquitectura VAE en caso de ejecución aislada
        class VAE_Fallback(torch.nn.Module):
            def __init__(self, input_dim=400, latent_dim=2):
                super().__init__()
                self.latent_dim = latent_dim
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, latent_dim * 2)  # mu y logvar juntos
                )
                self.decoder = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, input_dim),
                    torch.nn.Sigmoid()
                )
                
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
                
            def forward(self, x):
                h = self.encoder(x)
                mu, logvar = torch.chunk(h, 2, dim=-1)
                z = self.reparameterize(mu, logvar)
                x_hat = self.decoder(z)
                return x_hat, mu, logvar
                
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return VAE_Fallback(input_dim=400, latent_dim=latent_dim).to(device)


def entrenar_vae(modelo, datos_train, epochs=150, batch_size=8, lr=1e-3, kl_beta=0.005):
    """
    Entrena la red VAE desde cero utilizando el dataset de entrenamiento proporcionado.
    
    Parámetros:
      modelo (nn.Module): Modelo VAE inicializado.
      datos_train (np.ndarray): Dataset de entrenamiento de tamaño (N, 400).
      epochs (int): Número de épocas de entrenamiento.
      batch_size (int): Tamaño del lote.
      lr (float): Tasa de aprendizaje.
      kl_beta (float): Peso de la divergencia KL en la pérdida.
    """
    device = next(modelo.parameters()).device
    modelo.train()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=lr)
    
    dataset = LocalDataset(datos_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    
    for epoch in range(epochs):
        for _, histograms in loader:
            histograms = histograms.to(device)
            
            # Forward pass
            x_hat, mu, logvar = modelo(histograms)
            
            # Error de Reconstrucción (BCE) para evitar colapso posterior
            recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, histograms, reduction='sum') / histograms.size(0)
            
            # Divergencia KL
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            loss = recon_loss + kl_beta * kl_loss
            
            # Optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def ejecutar_deteccion_cambio_fase(N_list=[50, 100, 150, 200, 250, 300], epochs=150, use_real_simion=False):
    """
    Ejecuta el bucle de evaluación del cambio de fase (Grokking).
    Construye un dataset maestro vía LHS, evalúa la transmisión e histogamas,
    y entrena el VAE de manera justa con diferentes tamaños de dataset N.
    """
    print("\n--- Ejecutando Bucle de Detección del Cambio de Fase ---")
    n_val = 30
    n_max = max(N_list)
    total_samples = n_max + n_val
    
    # Límites físicos de los voltajes [-1000, 1000] V
    bounds_min = [-1000.0] * 8
    bounds_max = [1000.0] * 8
    
    # Generar muestras LHS (Módulo 1)
    df_voltajes = generar_muestras_lhs(total_samples, bounds_min, bounds_max, seed=42)
    voltages_array = df_voltajes.values
    
    # Evaluar todas las muestras en el simulador
    histograms_list = []
    transmissions_list = []
    
    for i in range(total_samples):
        volts = voltages_array[i]
        hist, trans = evaluar_simion(volts, use_real_simion=use_real_simion)
        histograms_list.append(hist)
        transmissions_list.append(trans)
        
    histograms = np.array(histograms_list)
    transmissions = np.array(transmissions_list)
    
    df_voltajes['transmision'] = transmissions
    
    # Separar en Set de Entrenamiento Maestro y Set de Validación Fijo
    train_histograms_master = histograms[:n_max]
    val_histograms_fixed = histograms[n_max:]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tensor = torch.tensor(val_histograms_fixed, dtype=torch.float32, device=device)
    
    validation_results = []
    
    for N in N_list:
        print(f"Evaluando N = {N:3d}...")
        
        # Tomar las primeras N muestras del dataset maestro
        train_subset = train_histograms_master[:N]
        
        # Inicializar el VAE desde cero (reiniciar pesos)
        modelo = iniciar_modelo_vae(latent_dim=2)
        
        # Entrenar VAE por E epochs
        entrenar_vae(modelo, train_subset, epochs=epochs, batch_size=8)
        
        # Calcular el Error de Reconstrucción (MSE) sobre el Set de Validación Fijo
        modelo.eval()
        with torch.no_grad():
            x_hat, _, _ = modelo(val_tensor)
            # Evaluamos usando MSE en la validación
            val_loss = torch.nn.functional.mse_loss(x_hat, val_tensor, reduction='mean').item()
            
        validation_results.append((N, val_loss))
        print(f"  -> Error de Reconstrucción (MSE) en Validación: {val_loss:.6f}")
        
    return validation_results, df_voltajes


# =============================================================================
#  MÓDULO 3: Visualización del Cambio de Fase
# =============================================================================
def encontrar_punto_codo(N_list, val_losses):
    """
    Encuentra el punto crítico Nc (codo) usando el método de distancia máxima
    entre la curva y la línea recta que une el primer y el último punto.
    """
    N_list = np.array(N_list)
    val_losses = np.array(val_losses)
    
    p1 = np.array([N_list[0], val_losses[0]])
    p2 = np.array([N_list[-1], val_losses[-1]])
    
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    distances = []
    for i in range(len(N_list)):
        p = np.array([N_list[i], val_losses[i]])
        v = p - p1
        proj = np.dot(v, line_vec_norm) * line_vec_norm
        dist = np.linalg.norm(v - proj)
        distances.append(dist)
        
    idx = np.argmax(distances)
    return N_list[idx]


def graficar_cambio_fase(resultados, path_guardado):
    """
    Grafica la curva de pérdida de validación vs. N y señala el codo Nc (transición de fase).
    """
    N_list, val_losses = zip(*resultados)
    nc = encontrar_punto_codo(N_list, val_losses)
    idx_nc = N_list.index(nc)
    val_nc = val_losses[idx_nc]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Graficar la pérdida de validación
    ax.plot(N_list, val_losses, color='#2b5c8f', linestyle='-', marker='o', linewidth=2.5, markersize=8, label='Error Validación (MSE)')
    
    # Línea punteada del punto crítico Nc
    ax.axvline(x=nc, color='#e74c3c', linestyle='--', linewidth=2.0, label=f'Punto Crítico $N_c$ ({nc} eval)')
    ax.scatter([nc], [val_nc], color='#e74c3c', s=180, zorder=5, edgecolor='black', linewidth=1.5, label='Elbow Point (Grokking)')
    
    # Títulos y Etiquetas
    ax.set_title('Bucle de Detección del Cambio de Fase (Phase Transition / Grokking)\nen el Aprendizaje del Gemelo Digital (VAE)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Tamaño del Dataset de Entrenamiento / Evaluaciones ($N$)', fontsize=11, labelpad=10)
    ax.set_ylabel('Error de Reconstrucción en Validación (MSE)', fontsize=11, labelpad=10)
    
    # Decoración y Grid
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=9.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper right')
    
    # Anotaciones de las fases
    y_range = max(val_losses) - min(val_losses)
    
    # Fase 1 Annotation
    ax.annotate('Fase 1: Memorizando Ruido\n(Overfitting local)', 
                xy=(nc - 25, val_losses[0] - y_range * 0.3), 
                xytext=(nc - 48, val_losses[0] - y_range * 0.75),
                arrowprops=dict(facecolor='#333', shrink=0.08, width=1, headwidth=6),
                fontsize=9.5, color='#444', fontweight='medium')
                
    # Fase 2 Annotation
    ax.annotate('Fase 2: Generalización Física\n(Comprensión del Espacio Latente)', 
                xy=(nc + 40, val_nc + y_range * 0.15), 
                xytext=(nc + 55, val_nc + y_range * 0.65),
                arrowprops=dict(facecolor='#333', shrink=0.08, width=1, headwidth=6),
                fontsize=9.5, color='#444', fontweight='medium')
                
    plt.tight_layout()
    os.makedirs(os.path.dirname(path_guardado), exist_ok=True)
    plt.savefig(path_guardado, bbox_inches='tight')
    plt.close()
    print(f"\n[Éxito] Gráfico guardado exitosamente en: {path_guardado}")


# =============================================================================
#  MÓDULO 4: Muestreo de Micro-Perturbación (Densificación de la Variedad)
# =============================================================================
def generar_perturbaciones_locales(mejores_voltajes, num_perturbaciones, sigma):
    """
    Genera nuevas combinaciones de voltajes sumando ruido Gaussiano alrededor
    de las mejores configuraciones para enriquecer localmente el dataset.
    
    Parámetros:
      mejores_voltajes (pd.DataFrame o np.ndarray): Las K mejores configuraciones de voltajes (K, 8).
      num_perturbaciones (int): Número de perturbaciones (M) a generar por cada base.
      sigma (float): Desviación estándar del ruido Gaussiano (ej. 20.0 V).
      
    Retorna:
      pd.DataFrame: DataFrame de forma (K * M, 8) con los nuevos voltajes perturbados.
    """
    if isinstance(mejores_voltajes, pd.DataFrame):
        # Excluir columnas no relacionadas a voltajes como 'transmision' si existe
        voltage_cols = [c for c in mejores_voltajes.columns if c.startswith('V')]
        base_voltages = mejores_voltajes[voltage_cols].values
    else:
        base_voltages = np.array(mejores_voltajes)
        voltage_cols = [f"V{k}" for k in [3, 6, 9, 10, 11, 12, 15, 18]]
        
    K, d = base_voltages.shape
    new_voltages = []
    
    for i in range(K):
        v_base = base_voltages[i]
        for _ in range(num_perturbaciones):
            # Sumar ruido Gaussiano N(0, sigma)
            ruido = np.random.normal(0, sigma, size=d)
            v_nuevo = v_base + ruido
            
            # Recortar voltajes para respetar límites físicos [-1000, 1000] V
            v_nuevo = np.clip(v_nuevo, -1000.0, 1000.0)
            new_voltages.append(v_nuevo)
            
    df_perturbed = pd.DataFrame(new_voltages, columns=voltage_cols)
    return df_perturbed


# =============================================================================
#  EJECUCIÓN PRINCIPAL (MAIN FLUX)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline de Recolección de Datos y Cambio de Fase.")
    parser.add_argument('--simion', action='store_true', help="Usar SIMION real en lugar del emulador sintético.")
    parser.add_argument('--epochs', type=int, default=150, help="Número de épocas para entrenar el VAE.")
    args = parser.parse_args()
    
    # 1. Asegurar la reproducibilidad del experimento
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=====================================================================")
    print("      SISTEMA DE MONITOREO Y RECOLECCIÓN PARA GEMELO DIGITAL        ")
    print("=====================================================================")
    print(f"Modo del simulador: {'[SIMION REAL]' if args.simion else '[SINTÉTICO / EMULADO]'}")
    
    # Definición de parámetros
    N_list = [50, 100, 150, 200, 250, 300]
    
    # 2. Correr el experimento de cambio de fase (Módulos 1 y 2)
    resultados, df_master = ejecutar_deteccion_cambio_fase(
        N_list=N_list, 
        epochs=args.epochs, 
        use_real_simion=args.simion
    )
    
    # 3. Graficar los resultados de pérdida de validación (Módulo 3)
    path_grafica = WORKSPACE_ROOT / "Figuras" / "phase_transition_grokking.png"
    graficar_cambio_fase(resultados, path_grafica)
    
    # 4. Muestreo de Micro-Perturbaciones (Módulo 4)
    print("\n--- Ejecutando Módulo 4: Densificación de la Variedad ---")
    
    # Filtrar las mejores configuraciones basadas en la transmisión del haz
    # En este caso, tomamos el top 5 combinaciones con transmisión mayor a un umbral (ej. 0.4)
    umbral_transmision = 0.4
    mejores_configs = df_master[df_master['transmision'] > umbral_transmision]
    
    # Si no hay suficientes por encima del umbral, tomamos simplemente el top 5
    if len(mejores_configs) < 5:
        mejores_configs = df_master.nlargest(5, 'transmision')
        
    print(f"Se seleccionaron las {len(mejores_configs)} mejores configuraciones para perturbación.")
    print("Detalles de las mejores combinaciones:")
    print(mejores_configs[['V3', 'V6', 'V9', 'V10', 'V11', 'V12', 'V15', 'V18', 'transmision']])
    
    # Generar M = 10 perturbaciones por cada base con sigma = 20.0 V
    M = 10
    sigma = 20.0
    df_perturbadas = generar_perturbaciones_locales(mejores_configs, num_perturbaciones=M, sigma=sigma)
    
    # Guardar los nuevos voltajes perturbados listos para SIMION
    path_salida_csv = WORKSPACE_ROOT / "Hackathon_student" / "voltajes_perturbados_lhs.csv"
    df_perturbadas.to_csv(path_salida_csv, index=False)
    
    print(f"\n[Éxito] Generadas {len(df_perturbadas)} nuevas combinaciones de voltajes perturbados.")
    print(f"Guardados en: {path_salida_csv}")
    print("Ejemplos de voltajes perturbados:")
    print(df_perturbadas.head())
    print("\nPipeline completado exitosamente.")
    print("=====================================================================")


if __name__ == "__main__":
    main()
