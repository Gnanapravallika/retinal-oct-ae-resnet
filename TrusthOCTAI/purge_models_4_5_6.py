"""
Purges any remaining references to Model 4, Model 5, and Model 6 across all notebooks and scripts.
Enforces strictly the 3 core models: resnet50, msf_resnet50, msf_cbam_resnet50.
"""
import json, os, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

target_files = [
    'notebooks/TrustOCT_Colab.ipynb',
    '../TrustOCT/NB4_Final_Analysis.ipynb',
    '../TrustOCT/NB3_Train_Evidential.ipynb',
    '../TrustOCT/NB5_Train_TrustOCT_Ablations.ipynb',
    '../TrustOCT/NB2_Train_Attention.ipynb'
]

for rel_path in target_files:
    full_path = os.path.join(os.path.dirname(__file__), rel_path)
    if not os.path.exists(full_path):
        continue
        
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        new_cells = []
        for cell in nb.get('cells', []):
            src = "".join(cell.get("source", []))
            # Skip cells containing references to models 4, 5, 6
            if any(m in src for m in ['msf_cbam_mixstyle', 'msf_cbam_edl', 'trustoct_expB', 'trustoct_expE', "'trustoct'"]):
                # If it's a code cell that ONLY trains models 4, 5, 6, remove it
                if "run_experiment('msf_cbam_mixstyle" in src or "run_experiment('msf_cbam_edl" in src or "run_experiment('trustoct" in src:
                    print(f"  - Removed training cell for models 4/5/6 in {rel_path}")
                    continue
                # If it's markdown title for models 4, 5, 6, remove it
                if "4. Train msf_cbam_mixstyle" in src or "5. Train msf_cbam_edl" in src or "6. Train trustoct" in src:
                    print(f"  - Removed markdown title for models 4/5/6 in {rel_path}")
                    continue
            new_cells.append(cell)
            
        nb['cells'] = new_cells
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
            
        print(f"[OK] Cleaned {rel_path}!")
    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
