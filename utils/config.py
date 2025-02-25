import os 

user = os.path.expanduser('~')
# Path to the directory where the data is stored
data_dir = os.path.join(user, 'projects/rrg-shahabkb/hamza97/data')
source_dirs = {"control": "Controls", "patients": "patients", }
bids_dir = os.path.join(data_dir, 'BIDS')
derivatives_dir = os.path.join(data_dir, 'derivatives')
sensors_to_keep = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "A1", "T3", "C3", "Cz",
                "C4", "T4", "A2", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

results_dir = os.path.join(data_dir, 'results')
n_subjects = 252
