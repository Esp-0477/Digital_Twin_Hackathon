import sys
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import subprocess

# Add Hackathon_student directory to path so we can import from optimizer.py
# __file__ is src/phase_1_VAE/dataset.py -> parent is src/phase_1_VAE -> parent.parent is src -> parent.parent.parent is workspace root
STUDENT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "Hackathon_student"
sys.path.append(str(STUDENT_DIR))

# Try importing the config from the student's optimizer script
try:
    import optimizer
    FIXED = optimizer.FIXED
    OPTIMIZE = optimizer.OPTIMIZE
    SIMION_INSTALL_DIR = optimizer.SIMION_INSTALL_DIR
    IOB_FILE = optimizer.IOB_FILE
    PA0_FILE = optimizer.PA0_FILE
    FLY_COMMAND = optimizer.FLY_COMMAND
    DETECTOR_REGION = optimizer.DETECTOR_REGION
except ImportError:
    # Fallback default values if import fails
    FIXED = {
        1: 500.0, 19: -2000.0, 2: 0.0,
        4: 0.0, 5: 0.0, 7: 0.0, 8: 0.0,
        13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0
    }
    OPTIMIZE = {3: (-1000.0, 1000.0), 6: (-1000.0, 1000.0), 9: (-1000.0, 1000.0),
                10: (-1000.0, 1000.0), 11: (-1000.0, 1000.0), 12: (-1000.0, 1000.0),
                15: (-1000.0, 1000.0), 18: (-1000.0, 1000.0)}
    SIMION_INSTALL_DIR = pathlib.Path(r"C:\Users\arnod\Documents\Workshop - MIT\In-site\SIMION")
    IOB_FILE = STUDENT_DIR / "SimpleSetUp.iob"
    PA0_FILE = STUDENT_DIR / "electrode_.PA0"
    FLY_COMMAND = f'simion --nogui fly --recording-output=out.txt --programs=0 --retain-trajectories=0 --restore-potential=0 "{IOB_FILE}"'
    DETECTOR_REGION = {"x": (70, 82), "y": (70, 83), "z": (403, 407)}


def get_positions_from_output(simion_output: str) -> np.ndarray:
    """
    Extracts the landing positions (x, y, z) of the ions from the SIMION stdout.
    """
    pattern = r"xyz\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)mm"
    matches = re.findall(pattern, simion_output)
    return np.array(matches, dtype=float)


def build_histogram(positions: np.ndarray, bins=20) -> np.ndarray:
    """
    Filters the positions to the detector region and constructs a 2D histogram
    representing the spatial distribution of the beam on the detector surface (x, y).
    
    The returned histogram is normalized by dividing by 500 (total number of launched ions),
    so that each bin value is in [0, 1] and the sum represents the transmission fraction.
    """
    if positions.size == 0:
        return np.zeros(bins * bins, dtype=np.float32)
        
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    
    # Filter points inside the 3D detector box
    x_min, x_max = DETECTOR_REGION["x"]
    y_min, y_max = DETECTOR_REGION["y"]
    z_min, z_max = DETECTOR_REGION["z"]
    
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    valid_pts = positions[mask]
    
    if valid_pts.size == 0:
        return np.zeros(bins * bins, dtype=np.float32)
        
    # Create 2D histogram on x and y axes
    hist, _, _ = np.histogram2d(
        valid_pts[:, 0], 
        valid_pts[:, 1], 
        bins=bins, 
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    # Apply Gaussian smoothing to make the profile less sparse and easier to reconstruct
    from scipy.ndimage import gaussian_filter
    hist = gaussian_filter(hist, sigma=0.8)
    
    # Normalize by 500 (the total number of ions launched)
    normalized_hist = hist / 500.0
    
    return normalized_hist.flatten().astype(np.float32)


def run_single_simulation(voltages: dict) -> np.ndarray:
    """
    Runs a single SIMION simulation for the given electrode voltages.
    Steps:
      1. Apply fastadj.
      2. Fly the ions.
      3. Extract landing coordinates.
    Returns:
      Array of landing coordinates (N, 3)
    """
    # Merge with fixed voltages
    all_volts = {**FIXED, **voltages}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    
    # Fastadj
    fastadj_cmd = f'simion --nogui fastadj "{PA0_FILE}" {settings}'
    subprocess.run(fastadj_cmd, cwd=str(SIMION_INSTALL_DIR), shell=True, check=True, stdout=subprocess.DEVNULL)
    
    # Fly
    fly_result = subprocess.run(
        FLY_COMMAND, 
        cwd=str(SIMION_INSTALL_DIR), 
        shell=True, 
        text=True, 
        capture_output=True, 
        check=True
    )
    
    # Get positions
    positions = get_positions_from_output(fly_result.stdout)
    return positions


class BeamlineDataset(Dataset):
    """
    PyTorch Dataset for the beamline emulator.
    Loads and normalizes:
      - Voltages (actions y in R^8), scaled to [-1, 1] range based on their optimization limits.
      - 2D histograms (states X in R^400) normalized to probability distributions (sum=1 for VAE)
        along with the transmission scalar.
    """
    def __init__(self, data_path: str):
        # Load npz file containing 'voltages' and 'histograms'
        data = np.load(data_path)
        self.raw_voltages = data['voltages'].astype(np.float32)       # shape: (N, 8)
        self.raw_histograms = data['histograms'].astype(np.float32)   # shape: (N, 400)
        
        # Apply Gaussian smoothing to raw histograms before any normalization
        from scipy.ndimage import gaussian_filter
        smoothed_raw_histograms = []
        for h in self.raw_histograms:
            h_2d = h.reshape(20, 20)
            h_smooth = gaussian_filter(h_2d, sigma=0.8)
            smoothed_raw_histograms.append(h_smooth.flatten())
        self.raw_histograms = np.array(smoothed_raw_histograms, dtype=np.float32)
        
        # Scale voltages to [-1, 1] using OPTIMIZE limits
        self.opt_keys = sorted(list(OPTIMIZE.keys()))  # [3, 6, 9, 10, 11, 12, 15, 18]
        self.min_vals = np.array([OPTIMIZE[k][0] for k in self.opt_keys], dtype=np.float32)
        self.max_vals = np.array([OPTIMIZE[k][1] for k in self.opt_keys], dtype=np.float32)
        
        # Min-max scale voltages to [-1, 1]
        self.voltages = 2.0 * (self.raw_voltages - self.min_vals) / (self.max_vals - self.min_vals) - 1.0
        
        # Calculate transmission fraction [0, 1] for each sample (sum of the raw histogram)
        self.transmissions = self.raw_histograms.sum(axis=1)
        
        # Normalize histograms to be true probability distributions (sum to 1.0)
        sum_val = self.raw_histograms.sum(axis=1, keepdims=True)
        # Avoid division by zero for zero-transmission samples
        self.histograms = self.raw_histograms / np.where(sum_val == 0.0, 1.0, sum_val)
        
    def __len__(self):
        return len(self.voltages)
        
    def __getitem__(self, idx):
        # Returns (voltages_tensor, probability_distribution_tensor, transmission_tensor)
        return (
            torch.tensor(self.voltages[idx], dtype=torch.float32),
            torch.tensor(self.histograms[idx], dtype=torch.float32),
            torch.tensor(self.transmissions[idx], dtype=torch.float32)
        )
        
    def denormalize_voltages(self, norm_volts):
        """
        Converts scaled voltages [-1, 1] back to their physical values.
        """
        # physical = (norm_volts + 1) * (max - min) / 2 + min
        if isinstance(norm_volts, torch.Tensor):
            min_t = torch.tensor(self.min_vals, device=norm_volts.device)
            max_t = torch.tensor(self.max_vals, device=norm_volts.device)
            return (norm_volts + 1.0) * (max_t - min_t) / 2.0 + min_t
        else:
            return (norm_volts + 1.0) * (self.max_vals - self.min_vals) / 2.0 + self.min_vals
