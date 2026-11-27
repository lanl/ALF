"""LoRA for HIPPYNN graphs: starts as the pretrained model, pickles, merges, and works under NTK."""

import copy
import io

import pytest
from ase import Atoms

torch = pytest.importorskip("torch")
pytest.importorskip("hippynn")

from hippynn import custom_kernels
from hippynn.graphs import GraphModule, inputs, networks, targets

from alframework.ml_interfaces.hippynn.compat import system_energy
from alframework.ml_interfaces.hippynn.lora import apply_lora, lora_parameter_count, merge_lora
from alframework.ml_interfaces.hippynn.ntk import EnergyForceNTK
from alframework.ml_interfaces.ntk import RowRequest, compute_rows


@pytest.fixture(scope="module", autouse=True)
def native_kernels():
    previous = custom_kernels.CUSTOM_KERNELS_ACTIVE
    custom_kernels.set_custom_kernels(False)
    yield
    custom_kernels.set_custom_kernels(previous)


@pytest.fixture(params=["Hipnn", "HipHopnn"])
def graph(request):
    with torch.random.fork_rng():
        torch.manual_seed(2)
        species, positions = inputs.SpeciesNode("species"), inputs.PositionsNode("positions")
        net = getattr(networks, request.param)("hipnn", (species, positions), module_kwargs={
            "possible_species": [0, 1, 8], "n_features": 6, "n_sensitivities": 4, "dist_soft_min": 0.5,
            "dist_soft_max": 2.5, "dist_hard_max": 3.0, "n_interaction_layers": 1, "n_atom_layers": 1})
        energy = system_energy(targets.HEnergyNode("henergy", net))
        return GraphModule((species, positions), (energy,)).double().eval()


WATER = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.03, 0.07], [-0.24, 0.93, -0.06]])


def evaluate(graph):
    return graph(torch.tensor([[8, 1, 1]]), torch.tensor(WATER.positions)[None])[0]


def test_lora_starts_as_the_pretrained_model_and_survives_pickling(graph):
    before = evaluate(graph)
    names = apply_lora(graph, rank=2, alpha=1.0)
    assert names and lora_parameter_count(graph) > 0
    torch.testing.assert_close(evaluate(graph), before)
    trainable = [n for n, p in graph.named_parameters() if p.requires_grad and p.is_floating_point()]
    network = [n for n in trainable if not n.endswith(("_lora_A", "_lora_B"))]
    assert network and all(n.startswith(tuple(f"moddict.{key}." for node, key in graph.names_dict.items()
                                              if node.name == "henergy")) for n in network)
    assert not any(n.endswith("_lora_base") for n in trainable)
    with torch.no_grad():
        for name, parameter in graph.named_parameters():
            if name.endswith("_lora_B"):
                parameter.normal_(0.0, 0.1)
    changed = evaluate(graph)
    assert not torch.allclose(changed, before)
    buffer = io.BytesIO()
    torch.save(graph, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    torch.testing.assert_close(evaluate(restored), changed)
    torch.testing.assert_close(evaluate(copy.deepcopy(graph)), changed)
    assert merge_lora(restored) > 0 and not any("_lora_" in n for n, _ in restored.named_parameters())
    torch.testing.assert_close(evaluate(restored), changed)


def test_ntk_rows_through_lora_hooks_agree_in_both_engines(graph):
    apply_lora(graph, rank=2)
    with torch.no_grad():
        for name, parameter in graph.named_parameters():
            if name.endswith("_lora_B"):
                parameter.normal_(0.0, 0.1)
    estimator = EnergyForceNTK(graph, [WATER.copy()], energy_scale=0.05, force_scale=0.1, sketch_dim=6)
    assert all(not n.endswith("_lora_base") for n in estimator.adapter.live_parameters())
    structure = estimator.structure_from_atoms(WATER)
    request = RowRequest(forces=True)
    _, forward = compute_rows(estimator.adapter, structure, request, estimator.sketch, engine="forward")
    _, reverse = compute_rows(estimator.adapter, structure, request, estimator.sketch, engine="reverse")
    for key in forward:
        torch.testing.assert_close(forward[key], reverse[key], rtol=1e-9, atol=1e-11)


def test_lora_refuses_unknown_targets(graph):
    with pytest.raises(ValueError, match="not nodes"):
        apply_lora(graph, targets=["nonesuch"])
