## What changed

## Risk
- [ ] Changes decision semantics
- [ ] Changes security/policy behavior
- [ ] Changes benchmark/eval behavior

## Evidence
- [ ] Unit tests
- [ ] Adversarial tests
- [ ] Benchmark smoke
- [ ] No secrets in fixtures/logs

Any change that alters thresholds/providers must report results on a frozen validation set and must not tune on the final test set.
