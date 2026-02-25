#!/usr/bin/env python3
"""Debug secondary structure detection issues."""

import numpy as np
from pathlib import Path
from Bio.PDB import PDBParser
import sys

sys.path.insert(0, str(Path(__file__).parent))
from features.geometric_features import parse_pdb_structure, compute_dihedral


def analyze_phi_psi(pdb_path):
    """Analyze phi/psi angles in a PDB file."""
    structure, residues_info = parse_pdb_structure(str(pdb_path))
    n_residues = len(residues_info)
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {pdb_path.name}")
    print(f"Residues: {n_residues}")
    print(f"{'='*60}")
    
    phi_psi_data = []
    
    for i in range(1, n_residues - 1):
        prev_res = residues_info[i - 1]
        curr_res = residues_info[i]
        next_res = residues_info[i + 1]
        
        # Check if we have all required atoms
        missing = []
        if prev_res['c_coord'] is None:
            missing.append("prev_C")
        if curr_res['n_coord'] is None:
            missing.append("curr_N")
        if curr_res['ca_coord'] is None:
            missing.append("curr_CA")
        if curr_res['c_coord'] is None:
            missing.append("curr_C")
        if next_res['n_coord'] is None:
            missing.append("next_N")
        
        if missing:
            print(f"  Residue {i+1} ({curr_res['aa']}): MISSING {missing}")
            continue
        
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        phi = compute_dihedral(
            prev_res['c_coord'],
            curr_res['n_coord'],
            curr_res['ca_coord'],
            curr_res['c_coord']
        )
        
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        psi = compute_dihedral(
            curr_res['n_coord'],
            curr_res['ca_coord'],
            curr_res['c_coord'],
            next_res['n_coord']
        )
        
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)
        
        # Classify region (matching geometric_features.py)
        if -100 <= phi_deg <= -30 and -80 <= psi_deg <= 0:
            region = "HELIX"
        elif (-180 <= phi_deg <= -60) and (90 <= psi_deg <= 180):
            region = "SHEET"
        elif (-180 <= phi_deg <= -60) and (-180 <= psi_deg <= -120):
            region = "SHEET"
        elif (-160 <= phi_deg <= -60) and (100 <= psi_deg <= 180):
            region = "SHEET"
        else:
            region = "coil"
        
        phi_psi_data.append({
            'resid': i + 1,
            'aa': curr_res['aa'],
            'phi': phi_deg,
            'psi': psi_deg,
            'region': region
        })
        
        print(f"  Res {i+1:2d} ({curr_res['aa']}): φ={phi_deg:7.1f}° ψ={psi_deg:7.1f}° → {region}")
    
    # Summary
    if phi_psi_data:
        helix_count = sum(1 for d in phi_psi_data if d['region'] == 'HELIX')
        sheet_count = sum(1 for d in phi_psi_data if d['region'] == 'SHEET')
        coil_count = sum(1 for d in phi_psi_data if d['region'] == 'coil')
        total = len(phi_psi_data)
        
        print(f"\n  Summary:")
        print(f"    Helix: {helix_count}/{total} ({100*helix_count/total:.1f}%)")
        print(f"    Sheet: {sheet_count}/{total} ({100*sheet_count/total:.1f}%)")
        print(f"    Coil:  {coil_count}/{total} ({100*coil_count/total:.1f}%)")
        
        # Check phi/psi distributions
        phis = [d['phi'] for d in phi_psi_data]
        psis = [d['psi'] for d in phi_psi_data]
        print(f"\n  Phi range: {min(phis):.1f}° to {max(phis):.1f}° (mean: {np.mean(phis):.1f}°)")
        print(f"  Psi range: {min(psis):.1f}° to {max(psis):.1f}° (mean: {np.mean(psis):.1f}°)")
        
        return phi_psi_data
    else:
        print("  No phi/psi angles computed (too short or missing atoms)")
        return []


def main():
    base_dir = Path(__file__).parent
    amp_dir = base_dir / "data" / "training_dataset" / "structures" / "AMP"
    decoy_dir = base_dir / "data" / "training_dataset" / "structures" / "DECOY"
    
    print("\n" + "🔍" * 30)
    print("  SECONDARY STRUCTURE DEBUG")
    print("🔍" * 30)
    
    # Analyze a few AMP structures (should be helical)
    print("\n\n=== AMP STRUCTURES (expected to be helical) ===")
    
    amp_files = sorted(amp_dir.glob("*.pdb"))[:5]
    for pdb_file in amp_files:
        analyze_phi_psi(pdb_file)
    
    # Analyze a few DECOY structures
    print("\n\n=== DECOY STRUCTURES ===")
    
    decoy_files = sorted(decoy_dir.glob("*.pdb"))[:3]
    for pdb_file in decoy_files:
        analyze_phi_psi(pdb_file)
    
    # Check if coordinates are in correct format
    print("\n\n=== COORDINATE CHECK ===")
    test_pdb = amp_files[0]
    structure, residues = parse_pdb_structure(str(test_pdb))
    
    print(f"\nFirst 3 residues of {test_pdb.name}:")
    for i, res in enumerate(residues[:3]):
        print(f"  Res {i+1} ({res['aa']}):")
        print(f"    N:  {res['n_coord']}")
        print(f"    CA: {res['ca_coord']}")
        print(f"    C:  {res['c_coord']}")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
