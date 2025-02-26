import os
import time
import pickle
import argparse
import logging

import mne
from mne_bids import BIDSPath
from goofi.data import to_data
from goofi.nodes.analysis.reveeeg import ReveEEG

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import derivatives_dir, results_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def segment_and_process(eeg_path, segment_duration=60):
    """
    Segment and process the EEG data to obtain embeddings.
    """
    try:
        raw = mne.io.read_raw_brainvision(eeg_path, preload=True)
    except Exception as e:
        logging.error(f"Error reading file {eeg_path}: {e}")
        return None

    # Limit processing to 20 minutes of data
    selected_duration = 20 * 60  
    total_duration = raw.times[-1]
    n_segments = int(selected_duration / segment_duration)
    embeddings = {}

    node = ReveEEG.create_standalone()
    node.setup()

    for seg in range(n_segments):
        tmin = seg * segment_duration
        tmax = min(tmin + segment_duration, total_duration)
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


def save_embeddings(embeddings, subject_id, segment_duration):
    """
    Save the embeddings for a single subject.
    """
    output_dir = os.path.join(results_dir, "embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"embeddings_{subject_id}_{segment_duration}.pkl"
    try:
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)
        logging.info(f"Embeddings saved to {output_file}")
    except Exception as e:
        logging.error(f"Could not save embeddings for {subject_id}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment and process EEG data.")
    parser.add_argument("--n_subjects", type=int, default=15, help="Number of subjects to process.")
    parser.add_argument("--start_subject", type=int, default=1, help="Subject ID to start processing.")
    parser.add_argument("--segment_duration", type=int, default=60, help="Duration of each segment in seconds.")
    return parser.parse_args()


def process_subject(subject_id, segment_duration):
    subject = f"sub-{subject_id}"
    embedding_path = os.path.join(results_dir, "embeddings", f"embeddings_{subject_id}_{segment_duration}.pkl")

    if embedding_path.is_file():
        logging.info(f"Embeddings already exist for {subject}. Skipping.")
        return

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

    logging.info(f"Processing {subject} using file {bids_path.fpath}")
    start_time = time.time()
    embeddings = segment_and_process(bids_path.fpath, segment_duration)
    if embeddings is not None:
        save_embeddings(embeddings, subject, segment_duration)
        logging.info(f"Processed {subject} in {time.time() - start_time:.2f} seconds")
    else:
        logging.error(f"Error processing {subject}")
    end_time = time.time()
    logging.info(f"Total time for {subject}: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")

def main():
    args = parse_args()
    logging.info(f"Processing {args.n_subjects} subjects starting from subject {args.start_subject}")
    for subject_id in range(args.start_subject, args.start_subject + args.n_subjects):
        process_subject(subject_id, args.segment_duration)


if __name__ == "__main__":
    main()