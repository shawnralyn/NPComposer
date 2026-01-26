# src/evaluation/

Evaluation metrics for generated molecules.

## metrics.py

Calculate validity, SA score, QED, and NP score for generated SMILES.

```bash
# Evaluate generated molecules
python metrics.py -i generated.txt

# Save results to JSON
python metrics.py -i generated.txt -o results.json
```

**Input:** Text file with one SMILES per line

**Output:**
```
Results:
  Valid: 950/1000 (95.0%)
  SA: 3.45 +/- 0.82
  QED: 0.52 +/- 0.18
  NP: 0.71 +/- 0.34
```

**Metrics:**
- `validity`: % of valid SMILES
- `sa_score`: Synthetic Accessibility (1-10, lower = easier)
- `qed`: Drug-likeness (0-1, higher = better)
- `np_score`: Natural Product likeness (higher = more NP-like)
