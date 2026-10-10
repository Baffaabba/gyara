## 1. Transcribe (ui.transcribe)
- fleurs_ha_ng_validation_10140605325461661233 (19.3 s, consent=True): asset 1, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10196261624983995568 (14.5 s, consent=True): asset 2, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10255951444507289726 (8.0 s, consent=True): asset 3, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10403460126806275266 (22.5 s, consent=True): asset 4, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10493053480703738801 (13.1 s, consent=True): asset 5, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10672551815457404310 (6.8 s, consent=True): asset 6, 1 segment(s); status card says: True
- fleurs_ha_ng_validation_10674933498692553496 (13.2 s, consent=False): asset 7, 1 segment(s); status card says: True

## 2. Correct and verify (ui.save_and_next)
- asset 1: draft 'dubbin nan shekarun da suka gabata, wani mutum da ake kira a...' -> verified 'Dubunnan shekarun da suka gabata, wani mutum da ake kira Ari...' status=verified
- asset 2: draft 'amazon kuma shine kogi mafi girma a duniya a wasu lokutan ni...' -> verified 'Amazon kuma shine kogi mafi fadi a Duniya, a wasu lokutan ni...' status=verified
- asset 3: draft 'a fankon lamarin gabinar bai zantan da ke gobe shi ta karfef...' -> verified 'A farkon lamari, kabilar Byzantine da ke gabashi ta karfafa ...' status=verified
- asset 4: draft 'a cikin wannan na'urar sufuri mai sauyawa, kowa na da alaka ...' -> verified 'A cikin wannan na’urar sufuri mai sauyawa kowa na da alaka d...' status=verified
- asset 5: draft 'yan'n'cike ta kuma ruwaito cewa masu na'antar samar da makam...' -> verified 'NHK ta kuma ruwaito cewa masana’antar samar da makamashin nu...' status=verified
- asset 6: draft 'ƙarfin barnansa ta shafi kowa daga sarki har na gama garin...' -> verified 'Karfin barnansa ta shafe kowa daga sarki har na gama-gari....' status=verified
- asset 7: draft 'sibiran gabashin afirka suna cikin tekun indiya kusa da gaba...' -> verified 'Tsibiran Gabashin Afirka suna cikin Tekun India kusa da gaɓa...' status=verified

## 3. Build dataset (ui.build_dataset, 'All files', held-out = FLEURS test)
- zip: E:/Repos/gyara/runs/smoke-finetune-export/workspace/exports/gyara-dataset-20261009-183309.zip (1584439 bytes)
- export dir: E:/Repos/gyara/runs/smoke-finetune-export/workspace/exports/gyara-dataset-20261009-183309
- files: ['README.md', 'audio/1f43713b03_00000.wav', 'audio/4fb7450166_00000.wav', 'audio/6676a8fb33_00000.wav', 'audio/b0731b854b_00000.wav', 'audio/c2b59fc67b_00000.wav', 'audio/ece68beeb9_00000.wav', 'manifest.jsonl', 'metadata.jsonl']
- app card: ['Training', 'dataset', 'ready.', '6', 'clips', '·', '1.0', 'minutes', '·', '0', 'speakers.', 'Held-out', 'check:', 'No', 'overlap', 'with', 'held-out', 'data.', 'Left', 'out:', '1', '×', 'no', 'recorded', 'speaker', 'consent']
- metadata.jsonl rows: 6; first: {"file_name": "audio/ece68beeb9_00000.wav", "transcription": "Dubunnan shekarun da suka gabata, wani mutum da ake kira Aristarchus yace Tsarin Rana ya zagaya Rana.", "duration": 11.052, "source": "gyara"}

## 4. datasets.load_dataset('audiofolder')
- DatasetDict({
    train: Dataset({
        features: ['audio', 'transcription', 'speaker', 'dialect', 'duration', 'asset_sha256', 'draft', 'source'],
        num_rows: 6
    })
})
- row 0: transcription='Dubunnan shekarun da suka gabata, wani mutum da ake kira Aristarchus yace Tsarin', sampling_rate=16000, samples=176832

## 5. gyara finetune on the export (NCAIR1/Hausa-ASR, 2 steps, CPU, eval.after false)
- exit 0 in 195 s (full output: finetune-output.txt)
- summary.json: {
  "train_utterances": 4,
  "dev_utterances": 2,
  "train_hours": 0.009533333333333333,
  "own_hours": 0.009533333333333333,
  "global_step": 2,
  "train_loss": 1.4806544780731201,
  "dev_wer": 0.075,
  "encoder_frozen": true,
  "dev_split": "random split of the training pool by utterance (2 utterances): no speaker ids, so the same voice can be in train and dev and dev WER is optimistic",
  "leakage_note": "Leakage check passed for audio: no file is shared (sha256 of every file). Speaker overlap could only be checked where both sides carry speaker ids: 6 of 6 training/dev and 621 of 621 held-out utterances have none. FLEURS publishes no speaker ids; its dataset card states that train speakers differ from dev/test speakers, which we rely on but cannot check.",
  "notes": [
    "language/task tokens: <|hausa|> <|transcribe|>"
  ],
  "before_after": null
}
- weights written: ['README.md', 'added_tokens.json', 'config.json', 'generation_config.json', 'merges.txt', 'model.safetensors', 'normalizer.json', 'preprocessor_config.json', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'training_args.bin', 'vocab.json']

Total 320 s
