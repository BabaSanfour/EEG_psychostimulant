#!/usr/bin/env python3
"""
Script to read subject EEG and metadata files, convert them to BIDS format,
and update the participants.tsv file.
"""

import os
import sys
import glob
import re
import logging
from typing import List, Set

import mne
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, source_dirs, bids_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_subject_ids(source_dir: str) -> List[str]:
    """
    Scan the source directory and return a sorted list of unique subject IDs.
    Assumes each file or folder name begins with a subject ID followed by a dot.
    """
    subject_ids: Set[str] = set()
    for item in os.listdir(source_dir):
        match = re.match(r"^([A-Z]{2}\d{5,6}[A-Z]?)\.", item)
        if match:
            subject_ids.add(match.group(1))
        else:
            parts = item.split('.')
            if parts and parts[0]:
                subject_ids.add(parts[0])
    subject_ids.discard('DskUUID')
    subject_ids.discard('')
    return sorted(subject_ids)


def read_subject_data(subject_id: str, source_dir: str, bids_root: str, mapping_df: pd.DataFrame):
    """
    Read files for a single subject, convert EEG data to BIDS format,
    and update participants.tsv with subject metadata.
    """
    pnt_file = os.path.join(source_dir, f"{subject_id}.pnt")
    if not os.path.exists(pnt_file):
        logging.error("No .pnt file found for %s", subject_id)
        raise RuntimeError(f"No .pnt file found for {subject_id}")

    with open(pnt_file, "rb") as f:
        subject_info = f.read()

    decoded_data = subject_info.decode("ISO-8859-1", errors="ignore")
    cleaned_data = decoded_data.replace("\x00", "")

    match = re.search(r"ID(\d{1,8}(?:\.\d)?)[EN]", cleaned_data)
    if not match:
        logging.error("Could not find subject ID in %s", subject_id)
        return

    new_subject_id = match.group(1)

    eeg_files = glob.glob(os.path.join(source_dir, f"{subject_id}.EEG"))
    logging.info("Found %d EEG files for subject %s", len(eeg_files), subject_id)
    if not eeg_files:
        logging.error("No EEG files found for %s", subject_id)
        raise RuntimeError(f"No EEG file found for {subject_id}")

    eeg_file_path = eeg_files[0]
    try:
        raw = mne.io.read_raw_nihon(eeg_file_path, preload=True)
    except Exception as e:
        raise RuntimeError(f"Error reading EEG file for {subject_id}: {e}")

    if new_subject_id in ['2.2', '999']:
        return

    if new_subject_id.endswith('.2') or new_subject_id.endswith('.1'):
        new_subject_id = new_subject_id[:-2]

    raw.info["line_freq"] = 60

    # Map subject IDs if necessary
    if len(new_subject_id) > 3:
        logging.info("Found subject ID: %s", new_subject_id)
        mapped = mapping_df[mapping_df["ID"] == int(new_subject_id)]
        if mapped.empty:
            logging.warning("Mapping not found for subject %s", new_subject_id)
            return
        new_subject_id = int(mapped["patient"].values[0])
        logging.info("Subject %s mapped to %s", subject_id, new_subject_id)

    new_subject_id = str(new_subject_id)
    bids_path = BIDSPath(
        root=bids_root,
        subject=new_subject_id,
        session="01",
        task="RESTING",
        run="01",
        suffix="eeg",
        extension=".vhdr",
    )

    write_raw_bids(
        raw,
        bids_path=bids_path,
        format="BrainVision",
        overwrite=True,
        allow_preload=True,
        verbose=False,
    )
    logging.info("Finished BIDS conversion for subject %s", subject_id)


def update_participants_tsv(bids_dir: str, subjects_df: pd.DataFrame):
    """
    Update the participants.tsv file in the BIDS directory with subject metadata.
    """
    tsv_path = os.path.join(bids_dir, "participants.tsv")
    participants_df = pd.read_csv(tsv_path, sep='\t')
    
    matched_ids = {id_: f"sub-{id_}" for id_ in subjects_df['ID'].values}
    subjects_df['participant_id'] = subjects_df['ID'].map(matched_ids)
    
    merged_df = pd.merge(participants_df, subjects_df, on="participant_id", how="left")
    merged_df.drop(columns=['age', 'ID', 'sex'], inplace=True, errors='ignore')
    merged_df.rename(columns={"Age": "age", "Sex": "sex"}, inplace=True)
    
    merged_df.to_csv(tsv_path, sep='\t', index=False)
    logging.info("Updated participants.tsv at %s", tsv_path)


def main():
    patient_dir = os.path.join(data_dir, source_dirs["patients"])
    control_dir = os.path.join(data_dir, source_dirs["control"])

    patient_ids = get_subject_ids(patient_dir)
    control_ids = get_subject_ids(control_dir)
    logging.info("Found %d patient IDs and %d control IDs", len(patient_ids), len(control_ids))
    
    mapping_file = os.path.join(data_dir, "csv", "match_missing_control.csv")
    mapping_df = pd.read_csv(mapping_file, header=None, names=["patient", "ID"], sep=';')
    
    subjects_file = os.path.join(data_dir, "csv", "subjects.csv")
    subjects_df = pd.read_csv(subjects_file)
    subjects_df.rename(columns={"Study ID": "ID", "Psychostimulant (y/n)": "group"}, inplace=True)
    
    for subject_id in patient_ids:
        try:
            read_subject_data(subject_id, patient_dir, bids_dir, mapping_df)
        except Exception as e:
            logging.error("Error processing patient %s: %s", subject_id, e)
    
    for subject_id in control_ids:
        try:
            read_subject_data(subject_id, control_dir, bids_dir, mapping_df)
        except Exception as e:
            logging.error("Error processing control subject %s: %s", subject_id, e)
    
    logging.info("Finished BIDS conversion for all subjects")
    logging.info("Subjects metadata head:\n%s", subjects_df.head())
    
    update_participants_tsv(bids_dir, subjects_df)


if __name__ == "__main__":
    main()