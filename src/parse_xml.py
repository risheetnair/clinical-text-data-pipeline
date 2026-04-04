import os
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Optional

RAW_DATA_DIR = "../data/raw"
MAX_FILES_PER_FOLDER = 50

def parse_medquad_file(file_path: str, source_folder: str) -> list[dict]:
    """
    Parse one XML file into a list of records.
    One QAPair = one row.
    """
    records = []

    tree = ET.parse(file_path)
    root = tree.getroot()

    document_id = root.attrib.get("id", "")
    source_attr = root.attrib.get("source", "")
    url = root.attrib.get("url", "")
    focus = root.findtext("Focus", default="")

    qa_pairs = root.find("QAPairs")
    if qa_pairs is None:
        return records

    for qa in qa_pairs.findall("QAPair"):
        question_el = qa.find("Question")
        answer_el = qa.find("Answer")

        question_text = question_el.text.strip() if question_el is not None and question_el.text else ""
        answer_text = answer_el.text.strip() if answer_el is not None and answer_el.text else ""

        qid = question_el.attrib.get("qid", "") if question_el is not None else ""
        qtype = question_el.attrib.get("qtype", "") if question_el is not None else ""
        pid = qa.attrib.get("pid", "")

        records.append({
            "source_folder": source_folder,
            "file_name": os.path.basename(file_path),
            "document_id": document_id,
            "source": source_attr,
            "url": url,
            "focus": focus,
            "pid": pid,
            "qid": qid,
            "qtype": qtype,
            "question": question_text,
            "answer": answer_text,
        })

    return records


def load_medquad_dataset(raw_data_dir: str = RAW_DATA_DIR,
                         max_files_per_folder: Optional[int] = MAX_FILES_PER_FOLDER) -> pd.DataFrame:
    """
    Walk through all subfolders under raw_data_dir, parse XML files,
    and combine them into one DataFrame.
    """
    all_records = []

    source_folders = sorted(
        folder for folder in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, folder))
    )

    for folder in source_folders:
        folder_path = os.path.join(raw_data_dir, folder)

        xml_files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith(".xml")
        )

        if max_files_per_folder is not None:
            xml_files = xml_files[:max_files_per_folder]

        print(f"Parsing {len(xml_files)} files from {folder}...")

        for filename in xml_files:
            file_path = os.path.join(folder_path, filename)

            try:
                file_records = parse_medquad_file(file_path, source_folder=folder)
                all_records.extend(file_records)
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    df = pd.DataFrame(all_records)
    return df


if __name__ == "__main__":
    df = load_medquad_dataset()

    print("\nFinal shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample rows:")
    print(df[["source_folder", "focus", "qtype", "question"]].head(10).to_string(index=False))