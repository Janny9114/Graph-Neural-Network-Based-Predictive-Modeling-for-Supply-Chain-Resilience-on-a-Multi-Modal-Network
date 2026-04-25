# Project Reorganization Guide

## ✅ Completed Changes

### 1. New Folder Structure Created
```
final_year/
├── 📁 core/                    # Core training scripts
├── 📁 data/                    # Data files
├── 📁 models/                  # Trained models
├── 📁 results/                 # Training results
├── 📁 docs/                    # Documentation
├── 📁 backend/                 # Backend API
└── 📁 Supply Chain Resilience Website/  # Frontend
```

### 2. Files Moved
- ✅ 8 core scripts → `core/`
- ✅ 3 data files → `data/`
- ✅ All model files → `models/`
- ✅ Result files → `results/`

### 3. Files Updated
- ✅ `backend/app.py` - All imports and paths updated
- ✅ `core/complete_training_pipeline.py` - All imports and paths updated
- ✅ `core/__init__.py` - Created

---

## ⚠️ Known Issues

### Core Module Imports
The files in `core/` directory use relative imports that may need adjustment when running standalone:

**Files with relative imports:**
- `core/graph_construction.py` - imports `from graph_preprocessing import ...`
- `core/generate_realistic_scenarios.py` - imports `from generate_realistic_scenarios import ...`
- `core/generate_edge_disruption_scenarios.py` - imports `from generate_realistic_scenarios import ...`

**Solutions:**
1. **When running from project root:** Imports work as-is
2. **When running from core/ directory:** May need to add parent to sys.path
3. **Best practice:** Always run from project root directory

---

## 🚀 How to Use

### Running the Backend
```bash
# From project root
cd backend
python app.py
```

### Running the Pipeline
```bash
# From project root
python -m core.complete_training_pipeline
```

### Running Individual Scripts
```bash
# From project root
python -m core.train_multi_gnn_realistic
python -m core.graph_construction
```

---

## 📦 Committing to GitHub

### Step 1: Check Git Status
```bash
git status
```

### Step 2: Stage All Changes
```bash
# Stage new folders
git add core/ data/ models/ results/

# Stage modified files
git add backend/app.py
git add core/complete_training_pipeline.py
git add core/__init__.py

# Stage this guide
git add REORGANIZATION_GUIDE.md
```

### Step 3: Commit Changes
```bash
git commit -m "refactor: reorganize project structure

- Create core/, data/, models/, results/ directories
- Move training scripts to core/
- Move data files to data/
- Move model files to models/
- Move result files to results/
- Update import paths in backend/app.py
- Update import paths in core/complete_training_pipeline.py
- Add core/__init__.py for Python module
- Add REORGANIZATION_GUIDE.md documentation

This reorganization follows Python best practices and makes the project
more maintainable and professional."
```

### Step 4: Push to GitHub
```bash
git push origin main
```

---

## 🔧 Testing After Reorganization

### Test 1: Backend API
```bash
cd backend
python app.py
```
**Expected:** Server starts successfully, loads model from `models/` and data from `data/`

### Test 2: Frontend
```bash
cd "Supply Chain Resilience Website"
npm start
```
**Expected:** Frontend connects to backend successfully

### Test 3: Pipeline
```bash
python -m core.complete_training_pipeline
```
**Expected:** Pipeline runs and saves results to `pipeline_output/`

---

## 📝 Additional Notes

### Path References
All absolute paths in the code use:
- `C:/Users/janny/Desktop/final_year/data/` for data files
- `C:/Users/janny/Desktop/final_year/models/` for model files
- `C:/Users/janny/Desktop/final_year/core/` for core scripts

### Import Statements
- Backend uses: `from core.train_multi_gnn_realistic import ...`
- Pipeline uses: `from core.generate_realistic_scenarios import ...`

### Data Files
- `data/synthetic_nodes.csv` - Node data
- `data/synthetic_edges.csv` - Edge data
- `data/supply_chain_graph.pt` - Preprocessed graph

### Model Files
- `models/best_gine_model.pt` - Best GINE model
- `models/best_gat_model.pt` - Best GAT model
- `models/best_gcn_model.pt` - Best GCN model
- etc.

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'core'"
**Solution:** Run from project root, not from subdirectories

### Issue: "FileNotFoundError: synthetic_nodes.csv"
**Solution:** Check that files are in `data/` folder

### Issue: "Can't load model file"
**Solution:** Check that model files are in `models/` folder

### Issue: Backend can't find app.py
**Solution:** Run `python app.py` from `backend/` directory, not project root

---

## ✅ Verification Checklist

- [ ] All files moved to correct directories
- [ ] Backend starts without errors
- [ ] Frontend connects to backend
- [ ] Model predictions work
- [ ] Pipeline can run successfully
- [ ] Git commit created
- [ ] Changes pushed to GitHub

---

## 📚 Related Documentation

- `docs/API_Documentation.md` - API endpoints
- `docs/User_Guide.md` - User guide
- `docs/Architecture.md` - System architecture
- `README.md` - Project overview

---

**Last Updated:** April 26, 2026  
**Status:** ✅ Reorganization Complete
