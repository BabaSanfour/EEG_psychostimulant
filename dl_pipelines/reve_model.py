import os
import mne
import pickle
import time
from goofi.data import to_data
from goofi.nodes.analysis.reveeeg import ReveEEG
import argparse
import sys
from mne_bids import BIDSPath
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import derivatives_dir, results_dir

def segment_and_process(eeg_path_file, segment_duration=60):
    """
    Segment and process the EEG data to obtain embeddings.
    """
    try: 
        raw = mne.io.read_raw_brainvision(eeg_path_file, preload=True)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    selected_duration = 20*60 # 20 min of data
    total_duration =  raw.times[-1]
    n_segments = int(selected_duration / segment_duration)
    embeddings = {}
    node = ReveEEG.create_standalone()
    node.setup()
    for seg in range(n_segments):
        tmin = seg * segment_duration
        tmax = tmin + segment_duration
        tmax = min(tmax, total_duration)
        raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)
        segment_data = raw_segment.get_data()
        embedding = node.process(
            to_data(
                segment_data,
                {
                    "sfreq": raw.info["sfreq"],
                    "channels": {"dim0": raw_segment.ch_names},
                },
            )
        )
        embeddings[seg] = embedding

    return embeddings

def save_embeddings(embeddings, subject_id):
    """
    Save the embeddings for a single subject.
    """
    output_file = os.path.join(results_dir, f"embeddings_{subject_id}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {output_file}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Segment and process EEG data.")
    argparser.add_argument("--n_subjects", type=int, default=15, help="Number of subjects to process.")
    argparser.add_argument("--start_subject", type=int, default=1, help="Subject ID to start processing.")
    argparser.add_argument("--segment_duration", type=int, default=60, help="Duration of each segment in seconds.")
    args = argparser.parse_args()
    n_subjects = args.n_subjects
    start_subject = args.start_subject
    segment_duration = args.segment_duration
    print(f"Processing {n_subjects} subjects starting from subject {start_subject}")
    for subject_id in range(start_subject, start_subject + n_subjects):
        subject = f"sub-{subject_id}"
        embedding_path = os.path.join(results_dir, "embeddings", f"embeddings_{subject}_{segment_duration}.pkl")
        # skip if embedding file exists
        if os.path.isfile(embedding_path):
            continue 
        bids_path = BIDSPath(
            root=derivatives_dir,
            subject=str(subject_id),
            session="01",
            task="RESTING",
            run="01",
            suffix="eeg",
            extension=".vhdr",
            datatype="eeg",
            processing="cleaned",
        )
        try:
            start_time = time.time()
            embeddings = segment_and_process(bids_path, segment_duration)
            with open(embedding_path, "wb") as f:
                pickle.dump(embeddings, f)
        except:
            print(f"Error subject {subject}")