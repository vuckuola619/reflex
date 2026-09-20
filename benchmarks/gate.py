from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser(description="Evidence gate for an Agent Reflex vs Jev claim on the same frozen holdout.")
    p.add_argument("report")
    p.add_argument("--accuracy-noninferiority",type=float,default=0.01,
                   help="maximum allowed accuracy deficit before the candidate fails")
    p.add_argument("--min-speedup",type=float,default=1.05)
    p.add_argument("--max-cost-ratio",type=float,default=1.0)
    p.add_argument("--ece-tolerance",type=float,default=0.02)
    args=p.parse_args()
    report=json.loads(Path(args.report).read_text())
    if "jev" not in report:
        raise SystemExit("report has no Jev comparison; no superiority claim is permitted")
    if report.get("jev_comparison_valid") is not True:
        errors=", ".join(report.get("jev_comparison_errors", [])) or "baseline calls incomplete"
        raise SystemExit("INVALID JEV COMPARISON: "+errors)
    if float(report["jev"].get("coverage", 0.0)) < 1.0:
        raise SystemExit("INVALID JEV COMPARISON: Jev baseline coverage must be 100% on the frozen holdout")
    provenance=report.get("provenance", {})
    observed=provenance.get("jev_observed_models") or []
    if not observed:
        raise SystemExit("INVALID JEV COMPARISON: observed Jev model version is missing")
    requested=provenance.get("jev_requested_model")
    if requested not in {None, "jev-latest", "jev-preview"} and requested not in observed:
        raise SystemExit(f"INVALID JEV COMPARISON: requested {requested} but observed {observed}")
    r,j=report["agent_reflex"],report["jev"]
    jev_latency=max(float(j["latency_p95_ms"]),1e-9)
    speedup=jev_latency/max(float(r["latency_p95_ms"]),1e-9)
    jev_cost=max(float(j["cost_usd"]),1e-12)
    cost_ratio=float(r["cost_usd"])/jev_cost if j["cost_usd"] else (0.0 if not r["cost_usd"] else float("inf"))
    checks={
        "accuracy_noninferior":r["overall_accuracy"]+args.accuracy_noninferiority>=j["overall_accuracy"],
        "selective_accuracy_noninferior":r["selective_accuracy"]+args.accuracy_noninferiority>=j["selective_accuracy"],
        "calibration_noninferior":r["ece"]<=j["ece"]+args.ece_tolerance,
        "required_p95_speedup":speedup>=args.min_speedup,
        "required_cost_ratio":cost_ratio<=args.max_cost_ratio,
        "irrelevant_context_noninferior":r["irrelevant_context_stability"]>=j["irrelevant_context_stability"],
        "prompt_injection_noninferior":r["prompt_injection_stability"]>=j["prompt_injection_stability"],
        "option_order_noninferior":r["option_order_stability"]>=j["option_order_stability"],
    }
    evidence={"checks":checks,"p95_speedup":speedup,"cost_ratio":cost_ratio,
              "paired_accuracy_delta":report.get("paired_accuracy_delta")}
    print(json.dumps(evidence,indent=2))
    failed=[k for k,v in checks.items() if not v]
    paired=report.get("paired_accuracy_delta")
    if paired and paired.get("ci95_low",-1)<-args.accuracy_noninferiority:
        failed.append("paired_bootstrap_accuracy_noninferiority")
    if failed:
        raise SystemExit("EVIDENCE GATE FAILED: "+", ".join(failed))
    print("PASS: configured evidence gates passed on this frozen paired benchmark. This claim does not generalize beyond the tested workload.")


if __name__=="__main__": main()
