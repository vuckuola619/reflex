# Live Jev versus Reflex comparison

Run completed 2026-09-21 local time (2026-09-20 23:16 UTC), on Windows 11 with Python 3.12.14. All 42 authorized Jev calls succeeded and reported model `jev-1.13.0`.

Both systems received the same 17 frozen command/HTTP observations and questions. These were recorded from the local repository and loopback services, not freshly executed for each provider. Correct answers come from exit codes or HTTP status, not Jev labels. Eight synthetic development cases and 17 prompt-injection replays are reported separately.

| Configuration | Correct / 17 | Abstentions | Wall p50 | Wall p95 | Estimated API cost / 1,000 |
|---|---:|---:|---:|---:|---:|
| Jev 1.13.0 | 17 | 0 | 781.84 ms | 962.65 ms | $0.02552 |
| Reflex exact evidence checks | 17 | 0 | 0.28 ms | 1.84 ms | $0 |
| Reflex local MiniLM semantic router | 1 | 16 | 170.56 ms | 208.10 ms | $0 |
| Reflex token-overlap baseline | 1 | 16 | 0.65 ms | 3.02 ms | $0 |

Local compute/electricity is excluded from API cost. Jev pricing is the adapter's configured $42 per billion input tokens, multiplied by returned usage; this is an estimate, not a billing statement. The 17 primary calls used 10,328 input tokens (estimated $0.000433776). All 42 calls used 23,616 input tokens (estimated $0.000991872).

## What the results establish

Reflex's exact checks matched Jev's answers with approximately 523 times lower p95 latency on this small workload. Those checks directly implement the predicates in the questions. This demonstrates the benefit of computing explicit facts locally, not general model superiority. Remote latency includes the network path from this machine; no conclusion about server-side inference speed follows.

Jev answered every observation, while the configured local semantic router could confidently answer only one. There were no wrong accepted semantic answers, but 16 abstentions count against overall completion. The local embedding setup is not a capable Jev replacement on these logs. It uses MiniLM revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 12 synthetic training fixtures and eight validation fixtures. Its configuration exactly matches the prior offline run; there was no holdout tuning between runs. Cold import/model loading took 12.40 seconds and is excluded from the per-request latency table.

## Prompt-injection replay

Each observation was repeated with an untrusted instruction to always report success. Jev and Reflex exact checks both remained correct on 17/17 and unchanged from their original answers. The local semantic router abstained on all 17. These appended-text tests do not establish general injection resistance, and replays are not additional independent observations.

Jev also answered the eight separate synthetic development cases correctly. These cases were not added to the primary accuracy denominator.

## Limits and provenance

The 17 observations are a correlated convenience sample from one repository. They include failed feature checks and fault-injection test output. With only 17 requests, the nearest-rank p95 is the maximum observed latency. Results are descriptive; they do not establish production percentiles or statistical equivalence. A bootstrap accuracy difference of zero for two perfect runs on this sample is not evidence of equal population accuracy.

Dataset SHA256: `226ce00fa0a11705fbcb17874121b0bfa041c907adb482403a81159c10b916c6`.

Exact approved payload SHA256 on disk: `bd1b5809dd5fa8c835b405657abca85aeed4d19bb4e3f32fb14601a30bbef9f0`. The previously recorded `f06bf7dbdb63cfb2cc36f4563a6eaab0bbc60c98edb96529abab8b8595aa13c2` is the same content with LF-normalized newlines; Windows wrote CRLF. Every outgoing request body was checked against the reviewed payload.

Raw per-case decisions, development results, stress results, frozen parameters and machine-readable report are in `artifacts/real-comparison-v2/`. No provider failures were counted as accuracy errors.

The supported next design is exact verification first, measured task-specific local routing second, and Jev fallback for judgments that remain uncertain. Better local semantic coverage requires a broader, independently labeled development dataset and a new untouched evaluation set.
