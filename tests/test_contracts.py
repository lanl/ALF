"""The contract checks plug-in authors run, run here on ALF's own built-ins, so the contracts match the code."""
from pathlib import Path

import numpy as np
import pytest
from parsl import python_app

from alframework.testing import ContractError, check_builder_task, check_engine, check_evaluator, check_labeler_task
from alframework.tools.config import load_config_file
from alframework.tools.molecules import MoleculesObject
from tests.helpers.fakes import spring_ensemble
from tests.helpers.molecules import SPRING_WATER, WATER

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
PROPERTIES = {"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]}


def _water_builder_config():
    folder = EXAMPLES / "simple_water"
    return load_config_file(str(folder / "builder_config.json"), str(folder) + "/")


@pytest.mark.parametrize("task", ["simple_condensed_phase_builder_task", "simple_multi_condensed_phase_builder_task"])
def test_the_built_in_builders_keep_the_contract(task):
    from alframework.builders import builders
    structures = check_builder_task(getattr(builders, task), builder_config=_water_builder_config(), count=2)
    assert [s.get_moleculeid() for s in structures] == ['mol-0000-0000000000', 'mol-0000-0000000001']


def test_a_builder_that_renames_its_structures_is_named():
    @python_app(executors=['alf_sampler_executor'])
    def renaming(moleculeid, builder_config):
        return MoleculesObject(WATER.copy(), 'my-own-id')

    with pytest.raises(ContractError, match="returned ID 'my-own-id', which wasn't asked for"):
        check_builder_task(renaming, builder_config={})


def test_the_ase_labeler_keeps_the_contract(tmp_path):
    from alframework.qm_interfaces.ase.generic import ase_calculator_task
    structures = [MoleculesObject(WATER.copy(), f'mol-0000-{index:010d}') for index in range(2)]
    labeled = check_labeler_task(ase_calculator_task, structures, QM_config={"ASE_calculator": "ase.calculators.emt.EMT"},
                                 properties_list=PROPERTIES, QM_scratch_dir=str(tmp_path))
    assert all(label.check_convergence() for label in labeled)


def test_a_labeler_whose_converged_labels_cant_be_stored_is_named():
    @python_app(executors=['alf_QM_executor'])
    def no_forces(molecule_objects, QM_config, properties_list):
        for molecule in molecule_objects:
            molecule.store_results({"energy": -1.0})
            molecule.set_converged_flag(True)
        return molecule_objects

    with pytest.raises(ContractError, match="converged label ALF can't store.*'forces'"):
        check_labeler_task(no_forces, [MoleculesObject(WATER.copy(), 'mol-0000-0000000000')], QM_config={},
                           properties_list=PROPERTIES)

    @python_app(executors=['alf_QM_executor'])
    def silent_failure(molecule_object, QM_config):
        molecule_object.set_converged_flag(False)
        return molecule_object

    with pytest.raises(ContractError, match="a failed label should say why"):
        check_labeler_task(silent_failure, [MoleculesObject(WATER.copy(), 'mol-0000-0000000000')], QM_config={},
                           properties_list=PROPERTIES)


def test_the_ase_ensemble_evaluator_keeps_the_contract():
    from alframework.ml_interfaces.ensemble_inference import ASEEnsembleEvaluator
    rows = check_evaluator(ASEEnsembleEvaluator(spring_ensemble()), [SPRING_WATER, WATER])
    assert len(rows) == 2 and rows[0]["Es"] > 0


def test_the_hippynn_batched_evaluator_keeps_the_contract(trained_ensemble):
    from alframework.ml_interfaces.hippynn.interface import load_hippynn_batched_ensemble
    rows = check_evaluator(load_hippynn_batched_ensemble(str(trained_ensemble), device="cpu"), [WATER, WATER])
    assert all(np.isfinite(row["Es"]) for row in rows)


class _Evaluator:
    n_models = 2

    def __init__(self, **changes):
        self.changes = changes

    def predict(self, atoms_list):
        prediction = {"energies": np.zeros((2, len(atoms_list))),
                      "forces": [np.zeros((2, len(atoms), 3)) for atoms in atoms_list]}
        prediction.update(self.changes)
        return prediction


@pytest.mark.parametrize("changes, message", [
    ({"energies": np.zeros((1, 2))}, r"not \(n_models, structures\)"),
    ({"forces": [np.zeros((2, 3, 3))]}, "one array per structure"),
    ({"energies": np.full((2, 2), np.nan)}, "nonfinite energies"),
    ({"uncertainty": [{"Es": 1.0}, {"Es": 1.0}]}, r"\['Es'\] are taken by built-in signals"),
    ({"uncertainty": [{"x": 1.0}]}, "one dict of named numbers per structure"),
])
def test_an_evaluator_that_breaks_the_layout_is_named(changes, message):
    with pytest.raises(ContractError, match=message):
        check_evaluator(_Evaluator(**changes), [WATER, WATER])


def test_an_evaluator_with_its_own_uncertainty_passes():
    rows = check_evaluator(_Evaluator(uncertainty=[{"dropout_std": 0.1}, {"dropout_std": 0.2}]), [WATER, WATER])
    assert [row["dropout_std"] for row in rows] == [0.1, 0.2]


@pytest.mark.parametrize("engine", ["ase", "torchsim", "tests.helpers.engines.CountingASEEngine"])
def test_the_built_in_engines_keep_the_contract(engine):
    from tests.helpers.engines import require_engine
    if engine == "torchsim":
        require_engine(engine)
    check_engine(engine, [SPRING_WATER, SPRING_WATER], spring_ensemble(), steps=3)


def test_an_engine_missing_methods_is_named():
    class Incomplete:
        def __init__(self, *args, **kwargs):
            pass

        def step(self, indices, n):
            return {}

    import sys
    sys.modules[__name__].Incomplete = Incomplete
    Incomplete.__qualname__ = "Incomplete"
    try:
        with pytest.raises(ContractError, match="The engine has no observe, atoms, finish"):
            check_engine(Incomplete, [SPRING_WATER], spring_ensemble())
    finally:
        del sys.modules[__name__].Incomplete
