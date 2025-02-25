import os
import sys
from mne_bids import BIDSPath, write_raw_bids
import mne
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import derivatives_dir, bids_dir, sensors_to_keep, n_subjects



def load_raw_data(subject_id):
    """
    Load the raw data for a single subject.
    """
    # Define the path to the raw data file
    bids_path = BIDSPath(
        root=bids_dir,
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
        datatype="eeg",
    )
    # Load the raw data
    raw = mne.io.read_raw_brainvision(bids_path, preload=True)
    return raw

def process_subject(raw):
    """
    Process the raw data for a single subject.
    """
    raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, misc=False)
    ch_names = raw.ch_names
    sensors_to_drop = [ch for ch in ch_names if ch not in sensors_to_keep]
    raw.drop_channels(sensors_to_drop)
    raw.notch_filter(60, filter_length="auto", phase="zero")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw

def save_derivative_raw(raw, subject_id):
    """
    Save the preprocessed raw data as a derivative.
    """
    bids_path = BIDSPath(
        root=derivatives_dir,
        subject=subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".fif",
        datatype="eeg",
        processing="cleaned",
    )
    write_raw_bids(
        raw,
        bids_path=bids_path,
        format="BrainVision",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )

    


if __name__ == "__main__":

    for subject_id in range(1, n_subjects + 1):
        subject_id = str(subject_id)
        print(f"\nProcessing subject {subject_id}")
        try:
            raw = load_raw_data(subject_id)
            raw = process_subject(raw)
            save_derivative_raw(raw, subject_id)
            print(f"Finished processing subject {subject_id}")
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            continue

