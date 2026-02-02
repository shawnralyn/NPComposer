# src/evaluation/

## metrics.py

Evaluate generated molecules.

```bash
python metrics.py -i generated.txt -o results.json
```

**Input:** Text file with one SMILES per line

**Output:**
```json
{
  "n_total": 1000,
  "n_valid": 950,
  "validity": 0.95,
  "sa_score": {"mean": 3.45, "std": 0.82},
  "qed": {"mean": 0.52, "std": 0.18},
  "np_score": {"mean": 0.71, "std": 0.34}
}
```

**Metrics:**
| Metric | Range | Target |
|--------|-------|--------|
| validity | 0-1 | > 0.95 |
| sa_score | 1-10 | < 4.0 |
| qed | 0-1 | > 0.5 |
| np_score | -3~+3 | > 0 |
