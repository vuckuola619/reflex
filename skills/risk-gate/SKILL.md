---
name: risk-gate
description: Gate agent actions by deterministic policy and calibrated risk, requiring review for irreversible, destructive, credential, or external-side-effect actions.
---
# Risk gate

Treat policy as stronger than model judgment. Block explicitly forbidden actions before semantic evaluation. For remaining actions, evaluate a bounded risk decision and use risk-specific thresholds: low-risk actions may auto-run at lower confidence; destructive or irreversible actions require higher confidence or human approval. Always preserve an audit record of the signals, decision, threshold, and provider.
