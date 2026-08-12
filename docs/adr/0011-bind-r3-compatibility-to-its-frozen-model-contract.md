# Bind R3 compatibility to its frozen model contract

Operator UI R3 model compatibility is determined by the frozen scorer contract—Python 3.11.15, NumPy 1.26.4, and exact fixed-fixture replay—not by libraries used by the Legacy Flask Prediction Interface. The observed scikit-learn 1.7.1 versus 1.9.0 warning is therefore not an R3 blocker because R3 loads a hash-bound JSON model and does not use scikit-learn for inference.
