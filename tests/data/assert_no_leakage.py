import pandas as pd

# Quick assertions to ensure we keep train/test clean and temporally ordered
tr_hist = pd.read_csv("tests/data/historical/train_form_guide.csv")
te_hist = pd.read_csv("tests/data/historical/test_form_guide.csv")
tr_race = pd.read_csv("tests/data/race_data/train_race_results.csv")
te_race = pd.read_csv("tests/data/race_data/test_race_results.csv")

# 1) Race ID disjointness
train_rids = set(tr_hist["race_id"].unique())
test_rids = set(te_hist["race_id"].unique())
assert train_rids.isdisjoint(test_rids), f"Race ID overlap found: {len(train_rids.intersection(test_rids))}"

# 2) Race-dog pairs disjointness
pairs_tr = set(zip(tr_hist.race_id, tr_hist.dog_id))
pairs_te = set(zip(te_hist.race_id, te_hist.dog_id))
assert pairs_tr.isdisjoint(pairs_te), f"Race-dog overlap found: {len(pairs_tr.intersection(pairs_te))}"

# 3) Temporal order: all train dates are earlier than any test date
tr_hist["race_datetime"] = pd.to_datetime(tr_hist["race_datetime"])
te_hist["race_datetime"] = pd.to_datetime(te_hist["race_datetime"])
assert tr_hist["race_datetime"].max() < te_hist["race_datetime"].min(), "Temporal split violated"

print("Synthetic data leakage checks passed:")
print(" - Train races:", len(train_rids))
print(" - Test races:", len(test_rids))

