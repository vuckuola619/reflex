from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, log
from statistics import mean
from typing import Any, Callable, Iterable

from agent_reflex.types import Decision


@dataclass(slots=True)
class EvalRow:
    gold: Any
    decision: Decision


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values=sorted(values); idx=min(len(values)-1,max(0,ceil(q*len(values))-1)); return values[idx]


def expected_calibration_error(rows: Iterable[EvalRow], bins: int=15) -> float:
    buckets=[[] for _ in range(bins)]
    for row in rows:
        d=row.decision
        if d.abstained: continue
        idx=min(bins-1,int(max(0,min(0.999999,d.confidence))*bins))
        buckets[idx].append((d.confidence,1.0 if d.value==row.gold else 0.0))
    total=sum(map(len,buckets))
    if not total: return 0.0
    return sum((len(b)/total)*abs(mean(x for x,_ in b)-mean(y for _,y in b)) for b in buckets if b)


def multiclass_brier(rows: Iterable[EvalRow]) -> float:
    scores=[]
    for row in rows:
        d=row.decision
        if d.abstained or not d.probabilities: continue
        labels=set(d.probabilities)|{str(row.gold)}
        scores.append(sum((d.probabilities.get(label,0.0)-(1.0 if label==str(row.gold) else 0.0))**2 for label in labels))
    return mean(scores) if scores else 0.0


def negative_log_likelihood(rows: Iterable[EvalRow]) -> float:
    vals=[]
    for row in rows:
        d=row.decision
        if d.abstained: continue
        p=d.probabilities.get(str(row.gold)) if d.probabilities else (d.confidence if d.value==row.gold else 1-d.confidence)
        p=min(1-1e-12,max(1e-12,float(p)))
        vals.append(-log(p))
    return mean(vals) if vals else 0.0


def summarize(rows: list[EvalRow]) -> dict[str,float]:
    non=[r for r in rows if not r.decision.abstained]
    lat=[r.decision.latency_ms for r in rows]
    correct=sum(r.decision.value==r.gold for r in non)
    total=len(rows)
    return {
        "n":float(total),
        "coverage":len(non)/total if total else 0.0,
        "selective_accuracy":correct/len(non) if non else 0.0,
        "overall_accuracy":correct/total if total else 0.0,
        "ece":expected_calibration_error(rows),
        "brier":multiclass_brier(rows),
        "nll":negative_log_likelihood(rows),
        "latency_p50_ms":_pct(lat,0.50),
        "latency_p95_ms":_pct(lat,0.95),
        "latency_p99_ms":_pct(lat,0.99),
        "cost_usd":sum(r.decision.cost_usd for r in rows),
        "cost_per_1k_usd":(sum(r.decision.cost_usd for r in rows)/total*1000) if total else 0.0,
        "abstention_rate":1-(len(non)/total if total else 0.0),
    }


def risk_coverage_curve(rows: list[EvalRow], thresholds: Iterable[float]=(0.5,0.6,0.7,0.8,0.9,0.95)) -> list[dict[str,float]]:
    out=[]
    for t in thresholds:
        selected=[r for r in rows if not r.decision.abstained and r.decision.confidence>=t]
        errors=sum(r.decision.value!=r.gold for r in selected)
        out.append({"threshold":float(t),"coverage":len(selected)/len(rows) if rows else 0.0,
                    "risk":errors/len(selected) if selected else 0.0})
    return out


def bootstrap_ci(rows: list[EvalRow], metric: Callable[[list[EvalRow]],float], *, iterations: int=1000, seed: int=17, alpha: float=0.05) -> tuple[float,float]:
    if not rows: return (0.0,0.0)
    rng=random.Random(seed); vals=[]; n=len(rows)
    for _ in range(iterations):
        sample=[rows[rng.randrange(n)] for _ in range(n)]
        vals.append(float(metric(sample)))
    vals.sort(); lo=vals[int((alpha/2)*iterations)]; hi=vals[min(iterations-1,int((1-alpha/2)*iterations))]
    return lo,hi


def paired_bootstrap_accuracy_delta(a: list[EvalRow], b: list[EvalRow], *, iterations: int=2000, seed: int=23) -> dict[str,float]:
    if len(a)!=len(b) or not a:
        raise ValueError("paired rows must be non-empty and equal length")
    rng=random.Random(seed); n=len(a); vals=[]
    for _ in range(iterations):
        idx=[rng.randrange(n) for _ in range(n)]
        ac=sum((not a[i].decision.abstained) and a[i].decision.value==a[i].gold for i in idx)/n
        bc=sum((not b[i].decision.abstained) and b[i].decision.value==b[i].gold for i in idx)/n
        vals.append(ac-bc)
    vals.sort()
    return {"mean":mean(vals),"ci95_low":vals[int(0.025*iterations)],"ci95_high":vals[min(iterations-1,int(0.975*iterations))]}
