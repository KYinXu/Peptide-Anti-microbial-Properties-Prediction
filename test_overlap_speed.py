import pandas as pd
import sys
from pathlib import Path
from collections import defaultdict

PAPER_CSV = "sequence_to_svm_minimal/data/proteomes/original_paper/Candidates Table 6 - Database of PDDPs.csv"
GENERATED_CSV = "sequence_to_svm_minimal/data/proteomes/pane_filtered_candidates/generated/final_candidates.csv"

print("Loading data...")
paper_df = pd.read_csv(PAPER_CSV)
paper_seqs = set(paper_df["Sequence"].dropna().unique())

gen_df = pd.read_csv(GENERATED_CSV)
gen_seqs = list(gen_df["sequence"].dropna().unique())

print("Finding missing sequences...")
# Near match logic
def is_near_match(p_seq, gen_seqs_set, max_delta=40):
    for g_seq in gen_seqs_set:
        if p_seq in g_seq and len(g_seq) - len(p_seq) <= max_delta:
            return True
        if g_seq in p_seq and len(p_seq) - len(g_seq) <= max_delta:
            return True
    return False

gen_seqs_set = set(gen_seqs)
missing = []
for p_seq in paper_seqs:
    if p_seq not in gen_seqs_set:
        if not is_near_match(p_seq, gen_seqs_set):
            missing.append(p_seq)

print(f"Found {len(missing)} missing sequences.")

print("Building 10-mer index...")
k = 10
index = defaultdict(list)
for i, g_seq in enumerate(gen_seqs):
    for j in range(len(g_seq) - k + 1):
        index[g_seq[j:j+k]].append(i)

print("Calculating max overlaps...")
def longest_common_substring(s1, s2):
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return longest

overlap_pcts = []
for p_seq in missing:
    # Find candidate g_seqs
    candidates = set()
    for j in range(len(p_seq) - k + 1):
        kmer = p_seq[j:j+k]
        candidates.update(index.get(kmer, []))
    
    max_overlap = 0
    for idx in candidates:
        g_seq = gen_seqs[idx]
        overlap = longest_common_substring(p_seq, g_seq)
        if overlap > max_overlap:
            max_overlap = overlap
            
    # If no 10-mer matched, fallback to full search (rare, but possible if overlap < 10)
    if not candidates:
        for g_seq in gen_seqs:
            overlap = longest_common_substring(p_seq, g_seq)
            if overlap > max_overlap:
                max_overlap = overlap
                
    overlap_pcts.append(max_overlap / len(p_seq) * 100)

print("Overlaps calculated!")
print(overlap_pcts[:10])
