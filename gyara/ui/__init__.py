"""Gyara's correction web app (Gradio). See ``gyara.ui.app``.

Imported lazily so ``import gyara.ui`` does not pull in Gradio until needed.
"""


def build_app(*args, **kwargs):
    from gyara.ui.app import build_app as _build

    return _build(*args, **kwargs)


def main(*args, **kwargs):
    from gyara.ui.app import main as _main

    return _main(*args, **kwargs)


__all__ = ["build_app", "main"]
