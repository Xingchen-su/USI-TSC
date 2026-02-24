# A Unified Semantic Interface for Traffic Signal Control in Heterogeneous Intersections

This repository contains scripts for training and evaluating a traffic signal control (TSC) model across heterogeneous intersection layouts using a unified semantic interface.

---

## 1. Before Training: Generate Phase-Matching Specifications

Before starting training, please generate the processed **phase-matching JSON** file by running:

```bash
python lane_dir/reorder_mask_builder.py
```

The generated JSON file will be saved to:

- `lane_dir/outputs_specs/`

### Notes
- During training, the framework **only uses the generated JSON** file.
- If needed, you may **manually adjust** the phase-matching results in the JSON to refine or correct the mapping.

### Using a Different Network File (Optional)
If you want to process a different road network, update the `net_file` variable in the main entry of `lane_dir/reorder_mask_builder.py` to the corresponding network file path.

---

## 2. Training

To train the model, run `run.py` with the desired `--scenario` and `--task`.

### Example: Train the `normal` scenario
```bash
python run.py --scenario normal --task regular
```

### Example: Train the `block` scenario
```bash
python run.py --scenario block --task block
```

### Notes on `--task block`
Setting `--task block` enables **random road blocking** during simulation.

⚠️ This option is only effective for the following rule-based scenarios:
- `normal`
- `hard`
- `block`

---

## 3. Evaluation

After training, model checkpoints are saved under:

- `ckpt/`

To evaluate a trained checkpoint:
1. Open `eva_sim.py` and locate the `main()` function.
2. Update the following paths as needed:
   - the checkpoint (weights) path
   - the network file path
3. Run:

```bash
python Eva_sim/eva_sim.py
```

---
