"""
Beamline voltage optimizer  —  STUDENT VERSION (you fill in the blanks)
======================================================================

This beamline is a small stack of metal electrodes that takes a beam of ions
(here: 500 singly-charged silicon ions, Si+ (mass 28 amu), at 15 eV) and tries
to steer and focus them all the way onto a Detector at the far end.

Each electrode can be set to a different *voltage*. Some are held fixed (the
source and detector, and the grounded parts), and the rest are free to choose.
There are too many combinations to try by hand, so we let the computer search:

    1. Optuna (an optimization library) PICKS a voltage for each free electrode.
    2. SIMION (ion-optics software) FLIES the 500 ions with those voltages.
    3. We SCORE the result: how many ions made it onto the Detector?
    4. Optuna learns from the score and picks better voltages next time.

================================  YOUR TASKS  ================================
Most of this program is written for you (reading SIMION's output, scoring the
result, checking the setup, all the configuration). TWO pieces are left for YOU,
marked with `TODO` and a `raise NotImplementedError(...)` that stops the program
until you finish them:

    TASK 1 -- the SIMION commands. Make Python tell SIMION what to do:
              (1a) the "fly" command string  ->  FLY_COMMAND
              (1b) the "fastadj" command      ->  apply_voltages()

    TASK 2 -- the Optuna loop. Drive the optimization:
              (2a) the objective() function
              (2b) building the study and running it, in main()

Search this file for "TODO" to jump to each one. A short SIMION command
reference is included just above Task 1; the full manual is the included
simion8manual.pdf (Appendix M, Command Line Interface), or simion.chm (see the
note there). Implement them in order (1a, 1b, 2a, 2b) and test as you go.

How positions are read (already done for you)
---------------------------------------------
SIMION prints lines like "xyz(12, 75, 77)mm" for each ion that reaches the
Detector; get_positions() below turns those into numbers.

What you need to run this
-------------------------
    * Python packages:   pip install optuna numpy
    * SIMION 8.1 installed, and the geometry already refined (Step 0).

Run it from a terminal with:   python optimizer.py
"""

import pathlib
import re
import subprocess

import numpy as np
import optuna
from optuna import Trial

# =============================================================================
#  CONFIG  (already filled in for you -- you may tweak these later)
# =============================================================================

# The folder this script is in, found automatically. Keep this script next to
# SimpleSetUp.iob and the electrode_.PA* files and it works on any computer.
SIM_FOLDER = pathlib.Path(__file__).resolve().parent

IOB_FILE = SIM_FOLDER / "SimpleSetUp.iob"   # the simulation to "fly"
PA0_FILE = SIM_FOLDER / "electrode_.PA0"    # the electrode model to adjust

# Where SIMION itself is installed (the normal Windows location). Only change
# this if SIMION lives somewhere else on your computer.
SIMION_INSTALL_DIR = pathlib.Path(r"C:\Users\arnod\Documents\Workshop - MIT\In-site\SIMION")

# Where to save your results (next to this script).
RESULTS_DB  = SIM_FOLDER / "beamline_study.db"
RESULTS_CSV = SIM_FOLDER / "beamline_results.csv"
STUDY_NAME  = "minimalsim_practice"

# How many attempts to run. Each one is a full SIMION simulation.
N_TRIALS = 150

#   "hits"   -> MAXIMIZE the number of ions that reach the Detector
#   "spread" -> MINIMIZE how spread-out the beam is at the Detector
OBJECTIVE = "hits"

# The electrodes we are allowed to tune, and their (physical!) voltage ranges.
#   electrode_number: (lowest_volts, highest_volts)
OPTIMIZE = {
    3:  (-1000.0, 1000.0),   # Einzel lens 1 (center)
    6:  (-1000.0, 1000.0),   # Einzel lens 2 (center)
    9:  (-1000.0, 1000.0),   # Quadrupole bender
    10: (-1000.0, 1000.0),   # Quadrupole bender
    11: (-1000.0, 1000.0),   # Quadrupole bender
    12: (-1000.0, 1000.0),   # Quadrupole bender
    15: (-1000.0, 1000.0),   # Voltage / deflection plate 1
    18: (-1000.0, 1000.0),   # Voltage / deflection plate 2
}

# The electrodes held at a fixed voltage (set on every run).
FIXED = {
    1:  500.0,     # HV source
    19: -2000.0,   # Detector
    2:  0.0,       # pipe
    4:  0.0, 5: 0.0, 7: 0.0, 8: 0.0,    # Einzel outer rings (grounded)
    13: 0.0, 14: 0.0, 16: 0.0, 17: 0.0, # ground plates
}

# Count an ion as a hit only if it lands inside this box (the Detector), in mm.
DETECTOR_REGION = {"x": (70, 82), "y": (70, 83), "z": (403, 407)}

STARTING_POINT = {
    "V3": -671.3954444976557,
    "V6": -172.96878240806473,
    "V9": 874.4469557832335,
    "V10": -409.74801424285454,
    "V11": 124.71246681522899,
    "V12": -843.4978757476558,
    "V15": 566.7452485578623,
    "V18": 663.3145025825927
}

# When a trial is invalid we return a deliberately terrible score.
BAD_SCORE = -1.0 if OBJECTIVE == "hits" else 1e9
DIRECTION = "maximize" if OBJECTIVE == "hits" else "minimize"

# =============================================================================
#  SIMION COMMANDS YOU WILL USE  (for Task 1)
# -----------------------------------------------------------------------------
#  You drive SIMION by running its command-line program. The sub-commands:
#
#    refine   solve the field from the geometry .PA#  ->  makes .PA0 + arrays.
#             You already ran this once in Step 0; the optimizer never calls it.
#
#    fastadj  set electrode voltages on the fast-adjust .PA0 array. Form:
#               simion --nogui fastadj "<PA0 file>" 1=500,3=120.5,19=-2000
#             (the voltages are a comma-separated list of  number=volts  pairs)
#
#    fly      fly the ions in the .iob through the field and print where they
#             land. Form:
#               simion --nogui fly --recording-output=out.txt --programs=0
#                      --retain-trajectories=0 --restore-potential=0 "<iob file>"
#
#  Flags:  --nogui (no window, required for scripting),
#          --recording-output=FILE (data file),
#          --retain-trajectories=0 (don't store full ion paths -> faster),
#          --restore-potential=0 (keep the voltages you just set),
#          --programs=0 (lean batch run).
#
#  MANUAL:  easiest is the included  simion8manual.pdf : everything is in
#    Appendix M (Command Line Interface); the commands are the same in that
#    older version. The newer manual is  simion.chm  in your SIMION folder
#    (next to simion.exe); search "command line". NOTE: .chm is a Windows Help
#    file and Windows OFTEN BLOCKS it (via Smart App Control, or the file's
#    "Mark of the Web"). If it opens blank or says "navigation cancelled",
#    right-click the file -> Properties -> tick "Unblock" -> OK. If Smart App
#    Control still blocks it, copy simion.chm to a local folder and open there.
# =============================================================================

# ----- TASK 1a: build the SIMION "fly" command string ------------------------
# Replace None with the full "fly" command (see the reference just above). It
# must fly IOB_FILE headless and write its recording to out.txt. Keep IOB_FILE
# wrapped in double quotes, because this folder's path contains a space:
#       f'... "{IOB_FILE}"'
FLY_COMMAND = f'simion --nogui fly --recording-output=out.txt --programs=0 --retain-trajectories=0 --restore-potential=0 "{IOB_FILE}"'


def run_simion(command: str) -> str:
    """Run one SIMION command-line call and return everything it printed.
    (Provided for you -- you do not need to change this.)

    It runs from inside the SIMION install folder so the `simion` program is
    found, and check=True makes a SIMION error raise so a bad trial is skipped.
    """
    result = subprocess.run(
        command,
        cwd=str(SIMION_INSTALL_DIR),
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return result.stdout


def apply_voltages(chosen: dict) -> None:
    """TASK 1b: build and run the SIMION 'fastadj' command to set the voltages.

    `chosen` is {electrode_number: volts} for the electrodes being optimized.

    Steps:
      1. Merge the FIXED electrodes with `chosen` so every electrode gets set:
             all_volts = {**FIXED, **chosen}
      2. Turn it into a comma-separated "number=volts" string, e.g.
             "1=500.0,2=0.0,3=120.5,...,19=-2000.0"
         Hint:  ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
      3. Build the fastadj command (see the reference above). Keep PA0_FILE in
         quotes:   f'simion --nogui fastadj "{PA0_FILE}" {settings}'
      4. Run it with  run_simion(command).
    """
    all_volts = {**FIXED, **chosen}
    settings = ",".join(f"{n}={v}" for n, v in sorted(all_volts.items()))
    command = f'simion --nogui fastadj "{PA0_FILE}" {settings}'
    run_simion(command)


def get_positions(simion_output: str) -> np.ndarray:
    """Pull the landing point of every reported ion out of SIMION's console text.
    (Provided for you.) Returns a table with one row per ion, columns [x, y, z].
    """
    pattern = r"xyz\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)mm"
    matches = re.findall(pattern, simion_output)
    return np.array(matches, dtype=float)


def _in_detector_region(positions: np.ndarray) -> np.ndarray:
    """True/False per ion: is it inside the DETECTOR_REGION box? (Provided.)"""
    x_min, x_max = DETECTOR_REGION["x"]
    y_min, y_max = DETECTOR_REGION["y"]
    z_min, z_max = DETECTOR_REGION["z"]
    return (
        (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
        (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max) &
        (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
    )


def _detector_hits(positions: np.ndarray) -> np.ndarray:
    """The ions that reached the Detector. (Provided.)"""
    if DETECTOR_REGION is not None:
        return positions[_in_detector_region(positions)]
    return positions


def count_hits(positions: np.ndarray) -> int:
    """Score = how many ions reached the Detector. Bigger = better. (Provided.)"""
    hits = _detector_hits(positions).shape[0]
    print(f"  ions on detector: {hits}")
    return hits


def beam_spread(positions: np.ndarray) -> float:
    """Score = spread of the beam at the Detector. Smaller = tighter. (Provided.)"""
    on_detector = _detector_hits(positions)
    if on_detector.shape[0] == 0:
        print("  no ions reached the detector")
        return BAD_SCORE
    spread = float(np.std(on_detector[:, 1]) + np.std(on_detector[:, 2]))
    print(f"  beam spread (y+z std): {spread:.3f}")
    return spread


def check_setup() -> None:
    """Make sure SIMION and the simulation files are ready. (Provided.)"""
    missing = [p for p in (SIMION_INSTALL_DIR, IOB_FILE) if not p.exists()]
    if missing:
        print("Could not find these required SIMION files/folders:")
        for p in missing:
            print(f"   - {p}")
        print("Keep this script in the same folder as the simulation files, and")
        print("check SIMION_INSTALL_DIR in the CONFIG section at the top.")
        raise SystemExit(1)
    if not PA0_FILE.exists():
        pa_define = SIM_FOLDER / "electrode_.PA#"
        simion_exe = SIMION_INSTALL_DIR / "simion.exe"
        print(f"The field file '{PA0_FILE.name}' does not exist yet.")
        print("You haven't refined the geometry. Run this once from this folder,")
        print("then run the optimizer again:")
        print(f'   & "{simion_exe}" --nogui refine "{pa_define.name}"')
        raise SystemExit(1)
    print("Found SIMION and the simulation files. Good to go!")


def objective(trial: Trial) -> float:
    """TASK 2a: one optimization attempt -- pick voltages, simulate, score it.

    Optuna calls this many times, each time giving you a fresh `trial`. The
    number you RETURN is the score Optuna tries to improve.

    Steps:
      1. For each electrode in OPTIMIZE, ask the trial for a voltage in its
         range, and collect them as chosen = {electrode_number: volts, ...}:
             for number, (low, high) in OPTIMIZE.items():
                 chosen[number] = trial.suggest_float(f"V{number}", low, high)
      2. Set the voltages and fly the ions. Wrap these two calls in a
         try / except subprocess.CalledProcessError; on error, return BAD_SCORE:
             apply_voltages(chosen)
             simion_output = run_simion(FLY_COMMAND)
      3. positions = get_positions(simion_output)
         If positions.shape[0] == 0:  return BAD_SCORE   (no ions reported)
      4. Return the score:
             count_hits(positions)   if OBJECTIVE == "hits"
             beam_spread(positions)  otherwise
    """
    chosen = {}
    for number, (low, high) in OPTIMIZE.items():
        chosen[number] = trial.suggest_float(f"V{number}", low, high)

    try:
        apply_voltages(chosen)
        simion_output = run_simion(FLY_COMMAND)
    except subprocess.CalledProcessError as e:
        print(f"SIMION error: {e}")
        return BAD_SCORE

    positions = get_positions(simion_output)
    if positions.shape[0] == 0:
        return BAD_SCORE

    if OBJECTIVE == "hits":
        return count_hits(positions)
    else:
        return beam_spread(positions)


def main() -> None:
    print("Beamline voltage optimizer (SIMION + Optuna) - STUDENT VERSION")
    print(f"Goal: {'MAXIMIZE ions on detector' if OBJECTIVE == 'hits' else 'MINIMIZE beam spread'}")
    print(f"Optimizing electrodes: {sorted(OPTIMIZE)}")

    check_setup()

    study = optuna.create_study(
        direction=DIRECTION,
        storage=f"sqlite:///{RESULTS_DB}",
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    if STARTING_POINT:
        study.enqueue_trial(STARTING_POINT)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    print("Optimization completed.")

    # ---- everything below is provided: it saves and reports your results -----
    try:
        study.trials_dataframe().to_csv(RESULTS_CSV, index=False)
        print(f"Saved all results to {RESULTS_CSV}")
    except Exception as error:
        print(f"Could not save the results CSV: {error}")

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"Trial number: {best.number}")
    print(f"Score:        {best.value}")
    print("Voltages:")
    for name, value in best.params.items():
        print(f"   {name}: {value}")


if __name__ == "__main__":
    main()
