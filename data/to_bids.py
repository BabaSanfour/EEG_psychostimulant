import os
import re
import sys
import glob
import logging
import mne
import pandas as pd
from mne_bids import write_raw_bids, BIDSPath

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.config import data_dir, source_dirs, bids_dir

logger = logging.getLogger(__name__)

def get_subject_ids(source_dir):
    """
    Scan the source directory and return a sorted list of unique subject IDs.
    We assume each file or folder name begins with a subject ID followed by a dot.
    """
    subject_ids = set()
    for item in os.listdir(source_dir):
        match = re.match(r"^([A-Z]{2}\d{5,6}[A-Z]?)\.", item)
        if match:
            subject_id = match.group(1)
            subject_ids.add(subject_id)
        else:
            parts = item.split('.')
            if parts and parts[0]:
                subject_ids.add(parts[0])
    subject_ids.discard('DskUUID')
    subject_ids.discard('')
    return sorted(subject_ids)

def read_subject_data(subject_id, source_dir, bids_root, mapping_df, subjects_df):
    """
    Read files related to a single subject, convert to BIDS format, and update participants.tsv
    with age and sex information from the subjects file.
    """
    eeg_files = glob.glob(os.path.join(source_dir, f"{subject_id}.EEG"))
    if not eeg_files:
        print(f"No EEG file found for {subject_id}")
        return
    eeg_file_path = eeg_files[0]
    try:
        raw = mne.io.read_raw_nihon(eeg_file_path, preload=True)
    except Exception as e:
        print(f"Error reading EEG file for {subject_id}: {e}")
        return
    raw.info["line_freq"] = 60

    pnt_file = os.path.join(source_dir, f"{subject_id}.pnt")
    if not os.path.exists(pnt_file):
        print(f"No .pnt file found for {subject_id}")
        return
    with open(pnt_file, "rb") as f:
        subject_info = f.read()
    decoded_data = subject_info.decode("ISO-8859-1", errors="ignore")
    cleaned_data = decoded_data.replace("\x00", "")
    match = re.search(r"ID(\d{1,8}(?:\.\d)?)[EN]", cleaned_data)
    if match is None:
        print(f"Could not find subject ID in {subject_id}")
        return
    new_subject_id = match.group(1)
    if len(new_subject_id) > 3:
        try: 
            mapped_value = mapping_df[mapping_df["patient"] == int(new_subject_id)]
            if mapped_value.empty:
                print(f"Mapping not found for subject {new_subject_id}")
                return
            new_subject_id = int(mapped_value["ID"].values[0])
            print(f"Found subject mapping: {subject_id} -> {new_subject_id}")
        except Exception as e:
            print(f"Error mapping subject ID {new_subject_id}: {e}")
            return
    if new_subject_id == '2.1':
        new_subject_id = 2
    if new_subject_id in ['2.2', '999']:
        return 0
    
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

    print(f"Finished BIDS conversion for subject {subject_id}")

if __name__ == "__main__":
    patient_dir = os.path.join(data_dir, source_dirs["patients"])
    control_dir = os.path.join(data_dir, source_dirs["control"])
    patient_ids = get_subject_ids(patient_dir)
    control_ids = get_subject_ids(control_dir)
    logger.info(f"Found {len(patient_ids)} patient IDs and {len(control_ids)} control IDs.")

    mapping_file = os.path.join(data_dir, "csv", "match_missing_control.csv")
    try:
        mapping_df = pd.read_csv(mapping_file)
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        mapping_df = pd.DataFrame()

    subjects_file = os.path.join(data_dir, "csv", "subjects.csv")
    try:
        subjects_df = pd.read_csv(subjects_file)
    except Exception as e:
        print(f"Error reading subjects file: {e}")
        subjects_df = pd.DataFrame()

    subjects_df.rename(columns={"Study ID": "ID", "Psychostimulant (y/n)": "group"}, inplace=True)

    for patient_id in patient_ids:
        read_subject_data(patient_id, patient_dir, bids_dir, mapping_df, subjects_df)
    for control_id in control_ids:
        read_subject_data(control_id, control_dir, bids_dir, mapping_df, subjects_df)
    print("Finished BIDS conversion for all subjects")
    print(subjects_df.head())

    tsv_file = os.path.join(bids_dir, "participants.tsv")
    tsv_file = pd.read_csv(tsv_file, sep='\t')
    ids = subjects_df['ID'].values
    matched_ids = {id: f"sub-{id}" for id in ids}
    subjects_df['participant_id'] = subjects_df['ID'].map(matched_ids)
    # group subject df and tsv file by participant_id
    tsv_file = pd.merge(tsv_file, subjects_df, on="participant_id", how="left")
    tsv_file.drop(columns=['age', 'ID', 'sex'], inplace=True)
    tsv_file.rename(columns={"Age": "age", "Sex": "sex"}, inplace=True)
    tsv_file.to_csv(os.path.join(bids_dir, "participants.tsv"), sep='\t', index=False)
    print(subjects_df.head())
    print(tsv_file.head())
