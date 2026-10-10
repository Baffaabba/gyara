# Sample files for the tester kit

`refs.txt` holds 5 reference sentences from the FLEURS Hausa test set.
`hyps.txt` holds what `NCAIR1/Hausa-ASR` wrote for the same 5 clips, in the
same order. Both come from `runs/hausa-asr-fleurs/predictions.jsonl`.

```bash
gyara score --ref docs/validation/samples/refs.txt --hyp docs/validation/samples/hyps.txt
```

With only 5 sentences the ranges are very wide. That is expected.

FLEURS is © Google, CC-BY-4.0.
