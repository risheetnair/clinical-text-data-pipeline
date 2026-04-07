import pandas as pd
import re
from parse_xml import load_medquad_dataset
import os

def clean_text(text: str) -> str:
    if not text:
        return ""

    # lower case
    text = text.lower()

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_symptoms(answer: str) -> str:
    NON_SYMPTOM_PHRASES = [
    "autosomal",
    "inheritance",
    "death in childhood",
    "human phenotype ontology",
    "hpo",
    "orphanet",
    "medlineplus medical dictionary",
    "approximate number of patients",
    "signs and symptoms approximate number of patients",
    ]
    
    if not answer:
        return ""

    text = answer.strip()

    # keep only content after common lead-in phrase, if present
    trigger_patterns = [
        r"Check with your doctor if you have any of the following:\s*",
        r"The signs and symptoms can include the following:\s*",
        r"Symptoms may include:\s*",
        r"Signs and symptoms may include:\s*",
    ]

    for pattern in trigger_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    # split on bullet-like separators
    parts = re.split(r"\s*-\s+|\u2022|\s{2,}", text)

    cleaned_parts = []
    for part in parts:
        part = part.strip(" .;:\n\t")
        part = re.sub(r"\s+", " ", part)

        # remove empty / tiny fragments
        if len(part) < 4:
            continue

        # remove generic non-symptom filler
        lower = part.lower()
        filler_phrases = [
            "check with your doctor",
            "these and other signs and symptoms",
            "signs and symptoms of",
            "the early signs and symptoms",
            "usually",
            "sometimes",
        ]
        if any(phrase in lower for phrase in NON_SYMPTOM_PHRASES):
            continue

        if any(phrase in lower for phrase in filler_phrases):
            continue

        if len(part.split()) > 12:
            continue

        cleaned_parts.append(part)

    # deduplicate while preserving order
    deduped = []
    seen = set()
    for item in cleaned_parts:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)

    # keep a manageable number of symptom phrases
    deduped = deduped[:6]

    return ", ".join(deduped)


def build_clinical_note(row):
    symptoms = extract_symptoms(row["answer"])

    if not symptoms:
        return ""

    return f"Pt presents with {symptoms}."


def main():
    df = load_medquad_dataset()

    # normalize + filter
    df["qtype"] = df["qtype"].str.lower().str.strip()
    df = df[df["qtype"] == "symptoms"].copy()

    # create clinical note
    df["clinical_note"] = df.apply(build_clinical_note, axis=1)

    # preview
    for i in range(5):
        print("\n---")
        print("Original:", df.iloc[i]["answer"][:300])
        print("Clinical:", df.iloc[i]["clinical_note"])


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_path = os.path.join(OUTPUT_DIR, "symptoms_clinical_notes.csv")

df = load_medquad_dataset()
df["qtype"] = df["qtype"].fillna("").str.lower().str.strip()
df = df[df["qtype"] == "symptoms"].copy()

df["clinical_note"] = df.apply(build_clinical_note, axis=1)

# optional: drop empty transformed notes
df = df[df["clinical_note"].str.strip() != ""].copy()

df.to_csv(output_path, index=False)

print(f"Saved processed dataset to: {output_path}")
print("Final shape:", df.shape)
print(df[["focus", "clinical_note"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()