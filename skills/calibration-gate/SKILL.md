---
name: calibration-gate
description: Calibrate decision confidence on a frozen validation split and choose risk-specific thresholds without touching the final test holdout.
---
# Calibration gate

Fit calibration only on labeled validation data. Report ECE, Brier/NLL, risk-coverage curves, coverage, selective accuracy, and high-risk slices. Tune thresholds by downstream action cost: auto-execution must require a higher bar than review or routing. Freeze the calibrator and thresholds before the final holdout. Recalibrate after any provider/model/version change.
