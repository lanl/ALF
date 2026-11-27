Fine-Tuning Excited States With X-MACE
======================================

`X-MACE <https://github.com/rhyan10/X-MACE>`__ adds excited states to a ground-state MACE model:

.. math::

   E_s = E_\mathrm{backbone} + \mathrm{scale}\cdot\mathrm{readout}_s + n_\mathrm{atoms}\cdot\mathrm{shift}_s

The readouts start at zero, so every state starts on the backbone's surface. The backbone can be a
ground-state foundation model, such as MACE-OFF, MACE-OMOL, or the MACE-MP family, or an earlier X-MACE
model. ALF fine-tunes it on excited-state labels ALF selects, for example GPU4PySCF TDA labels (see
:doc:`excited_states`). Each round starts every copy from the foundation model again, as for every family
in :doc:`fine_tuning`.

X-MACE installs as the package ``mace``, in place of mace-torch, so the ML and sampler workers need their
own environment with X-MACE in it. No pretrained excited-state X-MACE model is published; you supply the
ground-state foundation model, and its license applies.

.. code-block:: json

   {
     "fine_tune": {
       "adapter": "alframework.ml_interfaces.fine_tuning.xmace.XMaceFineTuner",
       "foundation_model": "foundation/mace-off23-small.model",
       "n_states": 4,
       "nacs_dataset": "nacr",
       "E0s": "estimated",
       "n_models": 1,
       "ntk": {"energy_scale": 0.05, "force_scale": 0.1}
     }
   }

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Setting
     - Meaning
   * - ``n_states``
     - The number of states S, read from ``E0`` … ``E{S-1}`` and ``F0`` … ``F{S-1}``. Or give
       ``energy_datasets`` and ``forces_datasets`` as lists.
   * - ``nacs_dataset``
     - Optional: the couplings stored as a ``pair_atomic`` property (GPU4PySCF's ``nacr``), with the
       shard's ``nac_pairs``. They're trained in X-MACE's smooth form, NAC × \|ΔE\|, which stays finite at
       crossings, for the pairs the labels have.
   * - ``E0s``
     - Required, as for MACE: isolated-atom energies at the labels' level of theory, a ``.json`` file,
       ``estimated``, or ``average``. Use ``foundation`` only for labels at the foundation model's level
       of theory. A level mismatch, such as CAM-B3LYP in implicit solvent against MACE-OFF's ωB97M-D3BJ in
       the gas phase, has to be absorbed by these references and X-MACE's per-state shifts. Check the
       first round's errors.
   * - ``model``
     - ``ExcitedMACE`` (default, independent per-state readouts), ``AutoencoderExcitedMACE`` (ordered
       states, smoother near crossings), or ``LEHExcitedMACE``.
   * - ``lora``, ``lora_rank``, ``lora_alpha``
     - Freeze the backbone and train low-rank adapters in it (defaults ``false``, 4, 1). The readouts and
       the adapter layer stay fully trainable. X-MACE merges the low-rank weights into the backbone when
       training ends.
   * - ``freeze``
     - Freeze the backbone's first N layers.
   * - ``replay_file``
     - Multi-head fine-tuning with replay of the foundation model's data, which limits forgetting.
   * - ``learning_rate``, ``max_num_epochs``, ``batch_size``
     - Defaults 5e-3, 15, and 10, from X-MACE's fine-tuning tutorial.
   * - ``valid_fraction``
     - Fraction of ALF frames reserved for validation, 0.1 by default. Each member's seed determines
       its split. ALF gives X-MACE separate training and validation files.
   * - ``xmace_options``
     - Other ``mace_run_train`` arguments. The defaults are the tutorial's loss weights: ``energy_weight``
       10, ``forces_weight`` 100, ``nacs_weight`` 10. Settings ALF manages are refused.
   * - ``ntk``
     - The NTK fit after training (see below), or ``null`` for none. Keys: ``energy_scale`` and
       ``force_scale`` (set from the model's validation errors), ``sketch_dim``, ``seed``,
       ``ridge_factor``, and ``parameters``. ``parameters`` is ``trainable``, ``excited``, or ``all``;
       ``excited``, the readouts and adapter only, is the default with LoRA.

Choosing frames with one model
------------------------------

Copies fine-tuned from one foundation model disagree very little, so ensemble disagreement is a weak
guide to what to label (see the warning in :doc:`fine_tuning`). After training, ALF therefore fits each
copy's NTK posterior on the frames it trained on, and saves it as ``ntk_basis.npz`` beside
``xmace.model``. The basis fit uses the same assigned device as training, recorded in ``ntk_fit.log``.
Sample with a copy's own uncertainty with
``"batched_calculator": "alframework.ml_interfaces.fine_tuning.xmace.load_xmace_ntk"``:

.. code-block:: json

   {
     "model_mode": "excited_state",
     "state_selection": {"mode": "fixed", "state": 1},
     "driver": {"engine": "torchsim"},
     "batched_calculator": "alframework.ml_interfaces.fine_tuning.xmace.load_xmace_ntk",
     "batched_calculator_options": {"scores": {"forces": false, "gap_pairs": [[1, 2]]}},
     "acquisition": {
       "terms": {"gap_ntk": {"uncertainty": "ntk_dE12", "cutoff": 0.05}},
       "eligibility": ["gap_ntk"],
       "strategy": "largest"
     }
   }

ALF chooses the propagation state using ``state_selection`` or the structure's state metadata. The
example cutoff is illustrative. The scores in :doc:`acquisition` include each state's energy, forces,
projected Hessian, gaps, smooth X-MACE outputs, and ``ntk_near_crossing_<state>`` flags.
For ensemble disagreement, load multiple copies with ``load_xmace_excited_state_ensemble``.

Each ``models/model-NNNN/model-NN`` holds ``xmace.model``, ``train.xyz``, ``valid.xyz``, and
``training_split.json``. The manifest records the ordered ALF frame IDs and file checksums for the
excited-state inputs. The NTK basis fits the recorded training frames. Preserve the manifest with
the checkpoint and NTK files when moving the model. Older ALF X-MACE directories without a recorded
split need retraining to produce a verified basis. Members also keep ``nac_pairs.json`` when
couplings are trained, along with X-MACE's ``train.log``, ``logs/``, ``results/``, and ``checkpoints/``.

On a CPU, a test fine-tunes a three-state ``ExcitedMACE`` with couplings from a small saved ground-state
MACE model, with and without LoRA. It then fits the NTK basis and loads the copy for packed MD; its
energies match the state-aware loader's.
