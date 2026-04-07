import pandas as pd
from parse_xml import load_medquad_dataset

df = load_medquad_dataset()

print("Original shape:", df.shape)

# Normalize qtype before analysis
df["qtype"] = df["qtype"].fillna("").str.lower().str.strip()

print("\nQtype counts:")
print(df["qtype"].value_counts())

# Length features for inspection
df["answer_char_len"] = df["answer"].fillna("").str.len()
df["question_char_len"] = df["question"].fillna("").str.len()

summary = (
    df.groupby("qtype")[["question_char_len", "answer_char_len"]]
      .agg(["count", "mean", "median", "min", "max"])
)

print("\nLength summary by qtype:")
print(summary)

for q in ["symptoms", "information", "treatment", "exams and tests"]:
    subset = df[df["qtype"] == q]
    print("\n" + "=" * 80)
    print(f"QTYPE: {q} | count={len(subset)}")
    print("=" * 80)

    for _, row in subset.head(2).iterrows():
        print("\nFOCUS:", row["focus"])
        print("QUESTION:", row["question"])
        print("ANSWER:", row["answer"][:500])

# Filter after inspection
symptoms_df = df[df["qtype"] == "symptoms"].copy()

print("\nFiltered shape (symptoms only):", symptoms_df.shape)
print("\nSymptoms sample:")
print(symptoms_df[["focus", "question"]].head(5).to_string(index=False))