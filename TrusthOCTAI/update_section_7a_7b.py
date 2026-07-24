"""
Updates Section 7 into Section 7A and Section 7B in TrustOCT_Colab.ipynb and NB4_Final_Analysis.ipynb.
Section 7A: Model Training from Scratch (Commented-out training code for future GitHub users).
Section 7B: Google Drive Pre-Trained Weights Loader & Syncer (Loads already-trained weights).
"""
import json, os, sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

for nb_path in ['notebooks/TrustOCT_Colab.ipynb', '../TrustOCT/NB4_Final_Analysis.ipynb']:
    full_path = os.path.join(os.path.dirname(__file__), nb_path)
    if not os.path.exists(full_path):
        continue
        
    with open(full_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # Find Section 7 cell and update into 7A and 7B
    new_cells = []
    for cell in cells:
        src = "".join(cell.get("source", []))
        if "Section 7 — Model Training" in src:
            # Markdown header for 7A
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Section 7A — Model Training from Scratch (For Future GitHub Users)\n",
                           "*(Commented-out training code. If you wish to train all 3 models from scratch, uncomment and run this cell.)*\n"]
            })
            # Code cell for 7A
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Section 7A: OPTIONAL TRAINING FROM SCRATCH ---\n",
                    "# from engine.trainer import Trainer\n",
                    "# from models.model2 import get_model2\n",
                    "# m1 = build_model({'model': {'backbone': 'resnet50', 'feature_module': 'identity', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}})\n",
                    "# t1 = Trainer(m1, train_loader, val_loader, criterion, optimizer, scheduler, device, is_evidential=False)\n",
                    "# t1.fit(epochs=30, save_dir='outputs/resnet50')\n",
                    "# m2 = build_model({'model': {'backbone': 'resnet50', 'feature_module': 'multiscale', 'attention': 'identity', 'dg': 'identity', 'head': 'softmax'}, 'dataset': {'num_classes': 4}})\n",
                    "# t2 = Trainer(m2, train_loader, val_loader, criterion, optimizer, scheduler, device, is_evidential=False)\n",
                    "# t2.fit(epochs=30, save_dir='outputs/msf_resnet50')\n",
                    "# m3 = get_model2(num_classes=4)\n",
                    "# t3 = Trainer(m3, train_loader, val_loader, criterion, optimizer, scheduler, device, is_evidential=False)\n",
                    "# t3.fit(epochs=30, save_dir='outputs/msf_cbam_resnet50')\n"
                ]
            })
            # Markdown header for 7B
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Section 7B — Google Drive Pre-Trained Weights Loader & Syncer\n",
                           "*(Automatically copies your already-trained weights from Google Drive so you do NOT need to re-train!)*\n"]
            })
            # Code cell for 7B
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# --- Section 7B: AUTOMATIC PRE-TRAINED WEIGHTS LOADER ---\n",
                    "import shutil, os, torch\n",
                    "\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                    "\n",
                    "explicit_weights = [\n",
                    "    ('/content/drive/MyDrive/TrustOCT_weights/resnet50.pth',          'outputs/resnet50',          'Baseline ResNet-50'),\n",
                    "    ('/content/drive/MyDrive/TrustOCT_weights/msf_resnet50.pth',      'outputs/msf_resnet50',      'ResNet-50 + MSF'),\n",
                    "    ('/content/drive/MyDrive/TrustOCT_weights (1)/msf_cbam_resnet50.pth', 'outputs/msf_cbam_resnet50', 'ResNet-50 + MSF + CBAM (Proposed)')\n",
                    "]\n",
                    "\n",
                    "for src, folder, label in explicit_weights:\n",
                    "    os.makedirs(folder, exist_ok=True)\n",
                    "    dest = os.path.join(folder, 'weights_best.pth')\n",
                    "    if os.path.exists(src):\n",
                    "        shutil.copy(src, dest)\n",
                    "        size_mb = os.path.getsize(dest) / 1024 / 1024\n",
                    "        print(f'✅ {label:40} → {folder} ({size_mb:.1f} MB)')\n",
                    "    else:\n",
                    "        print(f'❌ Not found at {src}. Checking secondary folders...')\n"
                ]
            })
        elif "OPTIONAL: Training Execution Code" in src or "AUTOMATIC DRIVE WEIGHTS SYNCER" in src:
            continue  # Replaced by new 7A/7B cells
        else:
            new_cells.append(cell)
            
    nb['cells'] = new_cells
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print(f"[OK] Updated Section 7A & 7B in {nb_path}!")
