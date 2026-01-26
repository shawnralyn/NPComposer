"""
SDF File Analyzer

Usage:
    python analyze_sdf.py -i coconut_sdf_3d.sdf
    python analyze_sdf.py -i coconut_sdf_3d.sdf -n 5000 -o report.txt
"""

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def analyze_sdf(sdf_path: str, n_samples: int = 1000):
    
    file_size = Path(sdf_path).stat().st_size / (1024 * 1024)
    print(f"File: {sdf_path} ({file_size:.1f} MB)")
    
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    
    total, valid = 0, 0
    properties = set()
    prop_examples = defaultdict(list)
    
    mol_weights, atom_counts, bond_counts = [], [], []
    ring_counts, rotatable, hbd, hba = [], [], [], []
    has_3d = []
    atom_types = Counter()
    samples = []
    
    print(f"Analyzing (max {n_samples} samples)...")
    
    for mol in suppl:
        total += 1
        
        if mol is None:
            continue
        valid += 1
        
        if valid <= n_samples:
            props = mol.GetPropsAsDict()
            for k, v in props.items():
                properties.add(k)
                if len(prop_examples[k]) < 3:
                    prop_examples[k].append(v)
            
            if mol.GetNumConformers() > 0:
                pos = mol.GetConformer().GetPositions()
                is_3d = not all(p[2] == 0 for p in pos)
                has_3d.append(is_3d)
            
            mol_weights.append(Descriptors.MolWt(mol))
            atom_counts.append(mol.GetNumAtoms())
            bond_counts.append(mol.GetNumBonds())
            ring_counts.append(rdMolDescriptors.CalcNumRings(mol))
            rotatable.append(Descriptors.NumRotatableBonds(mol))
            hbd.append(Descriptors.NumHDonors(mol))
            hba.append(Descriptors.NumHAcceptors(mol))
            
            for atom in mol.GetAtoms():
                atom_types[atom.GetSymbol()] += 1
            
            if len(samples) < 5:
                samples.append({
                    "smiles": Chem.MolToSmiles(mol),
                    "atoms": mol.GetNumAtoms(),
                    "mw": Descriptors.MolWt(mol)
                })
        
        if total % 10000 == 0:
            print(f"  {total:,} processed")
    
    print(f"\nTotal: {total:,}, Valid: {valid:,} ({100*valid/total:.1f}%)")
    
    print(f"\nProperties ({len(properties)}):")
    for p in sorted(properties):
        ex = ", ".join(str(e)[:30] for e in prop_examples[p])
        print(f"  {p}: {ex}")
    
    if has_3d:
        n3d = sum(has_3d)
        print(f"\n3D coords: {n3d}/{len(has_3d)} ({100*n3d/len(has_3d):.1f}%)")
    
    def stats(name, vals):
        if vals:
            print(f"  {name}: mean={statistics.mean(vals):.1f}, std={statistics.stdev(vals):.1f}, range=[{min(vals):.1f}, {max(vals):.1f}]")
    
    print("\nMolecular properties:")
    stats("MW", mol_weights)
    stats("Atoms", atom_counts)
    stats("Bonds", bond_counts)
    stats("Rings", ring_counts)
    stats("RotBonds", rotatable)
    stats("HBD", hbd)
    stats("HBA", hba)
    
    print("\nAtom types:")
    total_atoms = sum(atom_types.values())
    for sym, cnt in atom_types.most_common(10):
        print(f"  {sym}: {cnt:,} ({100*cnt/total_atoms:.1f}%)")
    
    print("\nSamples:")
    for i, s in enumerate(samples, 1):
        print(f"  [{i}] {s['smiles'][:60]}{'...' if len(s['smiles'])>60 else ''}")
        print(f"      atoms={s['atoms']}, MW={s['mw']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SDF file")
    parser.add_argument("-i", "--input", required=True, help="SDF file")
    parser.add_argument("-n", "--n_samples", type=int, default=1000, help="Samples to analyze")
    parser.add_argument("-o", "--output", help="Output file")
    
    args = parser.parse_args()
    
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    try:
        analyze_sdf(args.input, args.n_samples)
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = sys.__stdout__
            print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
