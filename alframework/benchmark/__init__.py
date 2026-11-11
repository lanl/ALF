"""Measure GPU use for ALF training and sampling, and suggest settings: ``python -m alframework.benchmark``.

Two kinds of measurement, each case in its own process so running out of memory ends only that case:

- **Training**: the time and peak GPU memory of one training step and one evaluation step of the
  HIPPYNN graph ALF trains, for each network profile and batch size. Labels are random, since speed
  and memory don't depend on their values.
- **Sampling**: the time and memory for an ensemble to evaluate energies and forces on a batch of
  structures, which is the work a sampler GPU does each MD step. Members are written untrained and
  loaded with ALF's own sampler loaders; untrained models cost the same as trained ones.

It runs out of the box on built-in structures, or reads a run directory's configs (``--run-dir``). QM
labeling isn't measured, nor is communication between GPUs when a member trains on several. Times
are for the model, not the whole MD step. See the "Sizing GPU Use" guide.
"""
