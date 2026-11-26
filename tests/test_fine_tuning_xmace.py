"""The X-MACE fine-tuning adapter: multi-state training data, X-MACE's command, and NTK after training."""

import json
import os

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from alframework.ml_interfaces.fine_tuning import (ENERGY_KEY, FORCES_KEY, SMOOTH_NACS_KEY, _label_targets,
                                                   shard_state_atoms)
from alframework.ml_interfaces.fine_tuning.xmace import XMaceFineTuner

N_FRAMES, N_STATES = 16, 3


def write_shard(path, frames=N_FRAMES):
    rng = np.random.default_rng(0)
    coordinates = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    coordinates = coordinates + rng.normal(scale=0.03, size=(frames, 3, 3))
    r = np.linalg.norm(coordinates[:, 1] - coordinates[:, 0], axis=1)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("water")
        group["species"] = np.array(["O", "H", "H"], dtype="S")
        group["coordinates"] = coordinates
        group["_id"] = np.array([f"frame-{i}" for i in range(frames)], dtype="S")
        for state in range(N_STATES):
            group[f"E{state}"] = -10.0 + 2.0 * state + (r - 0.96) ** 2 * (1 + state)
            group[f"F{state}"] = -0.1 * (state + 1) * (coordinates - coordinates.mean(1, keepdims=True))
        group["nacr"] = rng.normal(size=(frames, 1, 3, 3))
        group["nac_pairs"] = np.array([[1, 2]])
    return path


SETTINGS = dict(adapter="alframework.ml_interfaces.fine_tuning.xmace.XMaceFineTuner", foundation_model="f.model",
                n_states=N_STATES, nacs_dataset="nacr", E0s="average", n_models=1, seed=0, device="cpu")


def test_state_frames_carry_every_state_and_smooth_couplings(tmp_path):
    shard = write_shard(tmp_path / "data.h5")
    frames = shard_state_atoms([shard], [f"E{i}" for i in range(N_STATES)], [f"F{i}" for i in range(N_STATES)], "nacr")
    assert len(frames) == N_FRAMES
    with h5py.File(shard) as handle:
        group = handle["water"]
        energies = np.stack([group[f"E{i}"][()] for i in range(N_STATES)], 1)
        nacr = group["nacr"][()]
    for index, atoms in enumerate(frames):
        np.testing.assert_allclose(atoms.info[ENERGY_KEY], energies[index])
        assert atoms.arrays[FORCES_KEY].shape == (3, N_STATES, 3)
        gap = abs(energies[index, 2] - energies[index, 1])
        np.testing.assert_allclose(atoms.arrays[SMOOTH_NACS_KEY][:, 0], nacr[index, 0] * gap)
        assert atoms.info["nac_pairs"] == [[1, 2]]
    assert _label_targets({"n_states": 3}) == ("E0", "F0", None)
    assert _label_targets({"energy_datasets": ["S0", "S1"], "forces_datasets": ["G0", "G1"]}) == ("S0", "G0", None)


def test_settings_follow_the_tutorial_and_refuse_managed_arguments():
    tuner = XMaceFineTuner(SETTINGS)
    s = tuner.settings
    assert (s["learning_rate"], s["max_num_epochs"], s["batch_size"]) == (5e-3, 15, 10)
    assert s["energy_datasets"] == ["E0", "E1", "E2"] and s["ntk"]["parameters"] == "trainable"
    assert XMaceFineTuner({**SETTINGS, "lora": True}).settings["ntk"]["parameters"] == "excited"
    assert XMaceFineTuner({**SETTINGS, "ntk": None}).settings["ntk"] is None
    for bad, match in (({"E0s": None}, "E0s"), ({"xmace_options": {"lr": 1}}, "may not set"),
                       ({"model": "MACE"}, "model"), ({"ntk": {"sketch": 4}}, "ntk"), ({"n_states": 0}, "n_states")):
        settings = {**SETTINGS, **bad}
        if bad.get("E0s", 1) is None:
            settings.pop("E0s")
        with pytest.raises(ValueError, match=match):
            XMaceFineTuner(settings)
    with pytest.raises(ValueError, match="properties_list"):
        tuner.validate({"properties_list": {"E0": ["E0", "system", 1.0]}})


def test_training_data_and_command(tmp_path):
    shard = write_shard(tmp_path / "data.h5")
    member = tmp_path / "model-00"
    member.mkdir()
    tuner = XMaceFineTuner({**SETTINGS, "lora": True, "freeze": 2})
    data = tuner.prepare_data([shard], member)
    from ase.io import read
    frames = read(data, index=":")
    assert len(frames) == N_FRAMES and frames[0].arrays[FORCES_KEY].shape == (3, 3 * N_STATES)
    assert frames[0].arrays[SMOOTH_NACS_KEY].shape == (3, 3) and len(frames[0].info[ENERGY_KEY]) == N_STATES
    assert json.loads((member / "nac_pairs.json").read_text()) == [[1, 2]]
    command = tuner.command(data, str(member), 7, "cpu")
    for argument in ("--model=ExcitedMACE", "--n_energies=3", "--compute_nacs", "--nac_num=1",
                     f"--nacs_key={SMOOTH_NACS_KEY}", "--nacs_format=smooth", "--lora=True", "--lora_rank=4",
                     "--freeze=2", "--E0s=average", "--seed=7", "--forces_weight=100.0", "--multiheads_finetuning=False"):
        assert argument in command


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
def test_fine_tuning_then_ntk_end_to_end(tmp_path, lora):
    """Fine-tune a 3-state ExcitedMACE from a small ground-state MACE, then score with its NTK basis."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace.modules.excited", reason="X-MACE is not installed")
    from mace import modules
    from e3nn import o3
    from alframework.ml_interfaces.fine_tuning.xmace import load_xmace_excited_state_ensemble, load_xmace_ntk

    with torch.random.fork_rng():
        torch.manual_seed(0)
        block = modules.interaction_classes["RealAgnosticResidualInteractionBlock"]
        foundation = modules.ScaleShiftMACE(
            r_max=4.0, num_bessel=6, num_polynomial_cutoff=5, max_ell=2, interaction_cls=block,
            interaction_cls_first=block, num_interactions=2, num_elements=2, hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"), gate=torch.nn.functional.silu, atomic_energies=np.array([-1.0, -5.0]),
            avg_num_neighbors=2.0, atomic_numbers=[1, 8], correlation=2, atomic_inter_scale=1.0,
            atomic_inter_shift=0.0).double()
    torch.save(foundation, tmp_path / "foundation.model")
    shard = write_shard(tmp_path / "data.h5")
    member = tmp_path / "models" / "model-00"
    member.mkdir(parents=True)
    tuner = XMaceFineTuner({**SETTINGS, "foundation_model": str(tmp_path / "foundation.model"),
                            "max_num_epochs": 2, "batch_size": 4, "ntk": {"sketch_dim": 8}, "lora": lora})
    data = tuner.prepare_data([shard], member)
    tuner.train_member(data, member, 0, 0, "cpu", dict(os.environ))
    assert tuner.member_complete(member)
    evaluator = load_xmace_ntk(tmp_path / "models", "cpu", selected_state=1, scores={"hessian_probes": 2})
    estimator = evaluator.estimator
    assert estimator.state_names == ("E0", "E1", "E2") and estimator.training_structure_count == N_FRAMES
    names = list(estimator.adapter.live_parameters())
    # With LoRA, X-MACE merges the low-rank weights into the backbone, so NTK covers the readouts and adapter.
    assert any(name.startswith("backbone.") for name in names) != lora
    if lora:
        assert "LoRA" in (member / "train.log").read_text()
    atoms = shard_state_atoms([shard], ["E0", "E1", "E2"], ["F0", "F1", "F2"])[0]
    numbers, positions = torch.tensor(atoms.numbers), torch.tensor(atoms.positions)
    energies, forces = evaluator.evaluate(numbers, positions, None, torch.tensor(atoms.pbc), [3])
    observation = evaluator.observe_uncertainty(numbers, positions, None, torch.tensor(atoms.pbc))
    assert {"ntk_energy_E1", "ntk_hessian_E1", "ntk_dE12"} <= set(observation["uncertainty"])
    ensemble = load_xmace_excited_state_ensemble(tmp_path / "models", "cpu", dtype="float64")
    state_energies, state_forces = ensemble.evaluate_states(numbers, positions, None, torch.tensor(atoms.pbc), [3])
    assert state_energies.shape == (1, 3, 1) and state_forces.shape == (1, 3, 3, 3)
    np.testing.assert_allclose(energies[0].numpy(), state_energies[0, 1].numpy(), rtol=1e-5)
