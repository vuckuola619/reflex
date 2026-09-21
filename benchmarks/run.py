from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_reflex import DecisionKind, DecisionRequest, EvalRow, ProviderStage, ReflexEngine, summarize
from agent_reflex.providers import JevProvider, Prototype, PrototypeProvider, Rule, RuleProvider


def load(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def req(row):
    return DecisionRequest(
        state=row["state"],
        question=row["question"],
        kind=DecisionKind(row["kind"]),
        options=row.get("options", ()),
        risk=row.get("risk", 0.5),
    )


def build_engine(rows, with_jev: bool = False):
    # System 0: hard rules. Keep these narrow and auditable.
    rules = RuleProvider(
        [
            Rule("deny-root-delete", lambda r: "rm -rf /" in str(r.state), "deny"),
            Rule("deny-env-exfil", lambda r: ".env" in str(r.state) and "http" in str(r.state), "deny"),
            Rule("read-only-git", lambda r: str(r.state).startswith(("git status", "git diff")), "allow"),
        ]
    )
    train = [r for r in rows if r["split"] == "train"]
    proto = PrototypeProvider([Prototype(f"{r['question']}\n{r['state']}", r["gold"]) for r in train])
    stages = [ProviderStage(rules, threshold=1.0), ProviderStage(proto, threshold=0.58)]
    if with_jev:
        stages.append(ProviderStage(JevProvider(), threshold=0.70))
    return ReflexEngine(stages)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(Path(__file__).with_name("demo.jsonl")))
    p.add_argument("--with-jev", action="store_true", help="requires TYPESAFE_API_KEY and agent-reflex[jev]")
    args = p.parse_args()
    rows = load(Path(args.dataset))
    engine = build_engine(rows, args.with_jev)
    results = []
    for row in rows:
        if row["split"] != "test":
            continue
        d = engine.decide(req(row))
        results.append(EvalRow(row["gold"], d))
        print(
            json.dumps(
                {
                    "state": row["state"],
                    "gold": row["gold"],
                    "pred": d.value,
                    "provider": d.provider,
                    "confidence": round(d.confidence, 4),
                    "abstained": d.abstained,
                }
            )
        )
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    main()
