"""Fine-tuning Whisper Hausa models on human-verified Gyara exports.

Use ``gyara.finetune.train.train(config)`` (or ``python -m gyara.finetune.train
--config C``). It refuses to start when the training data shares audio or a
speaker with a held-out manifest. The submodule is deliberately not shadowed
by the function of the same name here.
"""
