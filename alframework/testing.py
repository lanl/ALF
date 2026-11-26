"""Contract checks for ALF plug-ins: run them in your own tests to learn that a builder, labeler, evaluator or
dynamics engine of yours fits ALF before a run finds out.

Each check calls your code in the test's own process, the way the coordinator or a worker would, and raises
:class:`ContractError` naming what doesn't fit. It returns what your code produced, for further checks of your own::

    from alframework.testing import check_builder_task, check_labeler_task

    def test_my_builder_builds_what_alf_needs():
        structures = check_builder_task(my_package.my_builder_task, builder_config={...})
        check_labeler_task(my_package.my_labeler_task, structures, QM_config={...},
                           properties_list={"energy": ["energy", "system", 1.0], "forces": ["forces", "atomic", 1.0]})

The checks cover what ALF relies on, not whether your chemistry is right.
"""
import inspect

import numpy as np

BUILT_IN_SIGNALS = ("Es", "Fs", "Fsmax", "Fsrms", "energy")
ENGINE_METHODS = ("step", "observe", "atoms", "finish", "set_temperature", "scale_cell", "attach_trajectory")


class ContractError(AssertionError):
    """A plug-in doesn't do what ALF relies on; the message says what."""


def _function(task):
    return getattr(task, 'func', task)


def _call(task, *sources):
    from alframework.tools.config import build_input_dict
    function = _function(task)
    try:
        inputs = build_input_dict(function, list(sources), raise_on_fail=True)
    except ValueError as exc:
        raise ContractError(f'{function.__name__} needs an input the check wasn\'t given: {exc}') from exc
    output = function(**inputs)
    from parsl.app.errors import RemoteExceptionWrapper
    if isinstance(output, RemoteExceptionWrapper):
        output.reraise()
    return output


def check_builder_task(task, *, builder_config, count=2, **inputs):
    """Build ``count`` structures, as the coordinator asks a builder for them, and check they can be sampled.

    A builder that takes ``moleculeids`` is asked for all of them in one call; one that takes ``moleculeid``, once
    per structure. ``inputs`` supplies anything else its signature names, such as ``master_directory``. Returns the
    structures.
    """
    from alframework.tools.builder_outputs import builder_outputs
    from alframework.tools.molecules import MoleculesObject
    parameters = inspect.signature(_function(task)).parameters
    ids = [f'mol-0000-{index:010d}' for index in range(count)]
    calls = [ids] if 'moleculeids' in parameters else [[one] for one in ids]
    structures = []
    for requested in calls:
        output = _call(task, {'moleculeid': requested[0], 'moleculeids': requested, 'builder_config': builder_config},
                       builder_config, inputs)
        try:
            built = builder_outputs(output, getattr(_function(task), '__name__', 'builder'))
        except TypeError as exc:
            raise ContractError(str(exc)) from exc
        if not built:
            raise ContractError(f'The builder returned no structures for {requested}.')
        structures.extend(built)
    for structure in structures:
        if not isinstance(structure, MoleculesObject):
            raise ContractError(f'The builder returned {type(structure).__name__}, not a MoleculesObject.')
        atoms = structure.get_atoms()
        if atoms is None or len(atoms) == 0:
            raise ContractError(f'{structure.get_moleculeid()}: the builder returned a structure without atoms.')
        if atoms.pbc.any() and not atoms.pbc.all():
            raise ContractError(f'{structure.get_moleculeid()}: partly periodic structures can\'t be stored; make it '
                                'fully periodic or not periodic.')
        if structure.get_moleculeid() not in ids:
            raise ContractError(f'The builder returned ID {structure.get_moleculeid()!r}, which wasn\'t asked for; keep '
                                'the IDs the coordinator gives.')
    return structures


def check_labeler_task(task, structures, *, QM_config, properties_list, **inputs):
    """Label ``structures`` as a QM worker would, and check every converged label can be stored.

    A labeler that takes ``molecule_objects`` labels them in one call (native batching); one that takes
    ``molecule_object``, one at a time. ``inputs`` supplies anything else its signature names, such as
    ``QM_scratch_dir``. A label that failed must be marked unconverged, with its reason in its metadata; a converged
    one must hold every property of ``properties_list`` with the right shape. Returns the labeled structures.
    """
    from alframework.tools.batching import flatten_molecule_output
    from alframework.tools.label_validation import validate_label
    parameters = inspect.signature(_function(task)).parameters
    common = {'QM_config': QM_config, 'properties_list': properties_list}
    if 'molecule_objects' in parameters:
        labeled = flatten_molecule_output(_call(task, {'molecule_objects': list(structures)}, common, inputs))
    elif 'molecule_object' in parameters:
        labeled = [result for structure in structures
                   for result in flatten_molecule_output(_call(task, {'molecule_object': structure}, common, inputs))]
    else:
        raise ContractError('A labeler takes molecule_objects (a batch) or molecule_object (one structure).')
    for label in labeled:
        if label.check_convergence():
            try:
                validate_label(label, properties_list)
            except ValueError as exc:
                raise ContractError(f'{label.get_moleculeid()}: a converged label ALF can\'t store: {exc}') from exc
        elif not any(key in label.get_metadata() for key in ('labeling_error', 'qm_error')):
            raise ContractError(f'{label.get_moleculeid()}: a failed label should say why, in its metadata\'s '
                                'labeling_error or qm_error.')
    return labeled


def _check_prediction(prediction, members, atoms_list, where):
    systems = len(atoms_list)
    energies = np.asarray(prediction.get('energies'))
    if energies.shape != (members, systems):
        raise ContractError(f'{where} energies have shape {energies.shape}, not (n_models, structures) = '
                            f'{(members, systems)}.')
    forces = prediction.get('forces')
    if not isinstance(forces, (list, tuple)) or len(forces) != systems:
        raise ContractError(f'{where} forces are a list with one array per structure.')
    for atoms, member_forces in zip(atoms_list, forces):
        if np.shape(member_forces) != (members, len(atoms), 3):
            raise ContractError(f'{where}: a structure of {len(atoms)} atoms has forces of shape '
                                f'{np.shape(member_forces)}, not (n_models, atoms, 3) = {(members, len(atoms), 3)}.')
    if not np.isfinite(energies).all() or not all(np.isfinite(f).all() for f in forces):
        raise ContractError(f'{where} returned nonfinite energies or forces.')
    own = prediction.get('uncertainty')
    if own is not None:
        if not isinstance(own, (list, tuple)) or len(own) != systems or not all(isinstance(row, dict) for row in own):
            raise ContractError(f'{where} uncertainty is a list with one dict of named numbers per structure.')
        clash = sorted({name for row in own for name in row} & set(BUILT_IN_SIGNALS))
        if clash:
            raise ContractError(f'Uncertainty names {clash} are taken by built-in signals; choose others.')
    from alframework.ml_interfaces.ensemble_inference import atomic_uncertainty_batch, ensemble_statistics
    try:
        atomic_uncertainty_batch(prediction.get('atomic_uncertainty'), [len(atoms) for atoms in atoms_list])
        return ensemble_statistics(prediction)
    except ValueError as exc:
        raise ContractError(str(exc)) from exc


def _evaluate_packed(evaluator, atoms_list):
    """``evaluate`` on the packed tensors an engine passes, as a ``predict``-style dict."""
    import torch
    from alframework.ml_interfaces.ensemble_inference import (add_observed_uncertainty, batched_uncertainty_observations,
                                                              merge_uncertainty_observations)
    if len({tuple(atoms.pbc) for atoms in atoms_list}) != 1:
        raise ContractError('Give check_evaluator structures with one periodicity; engines pack them that way.')
    counts = [len(atoms) for atoms in atoms_list]
    numbers = torch.as_tensor(np.concatenate([atoms.numbers for atoms in atoms_list]))
    positions = torch.as_tensor(np.concatenate([atoms.positions for atoms in atoms_list]), dtype=torch.float64)
    cells = torch.as_tensor(np.stack([atoms.cell.array for atoms in atoms_list]), dtype=torch.float64)
    pbc = torch.as_tensor(atoms_list[0].pbc)
    packed_energies, packed_forces = evaluator.evaluate(numbers, positions, cells, pbc, counts)
    energies, forces = (np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value)
                        for value in (packed_energies, packed_forces))
    if forces.ndim != 3 or forces.shape[1:] != (sum(counts), 3):
        raise ContractError(f'evaluate() forces have shape {forces.shape}, not (n_models, atoms in the batch, 3) = '
                            f'{(evaluator.n_models, sum(counts), 3)}.')
    offsets = np.cumsum([0, *counts])
    prediction = {'energies': energies,
                  'forces': [forces[:, start:end] for start, end in zip(offsets[:-1], offsets[1:])]}
    if getattr(evaluator, 'last_uncertainty', None) is not None:
        prediction['uncertainty'] = evaluator.last_uncertainty
    if getattr(evaluator, 'last_atomic_uncertainty', None) is not None:
        prediction['atomic_uncertainty'] = evaluator.last_atomic_uncertainty
    try:
        if callable(getattr(evaluator, 'observe_uncertainty_batch', None)):
            observations = batched_uncertainty_observations(
                evaluator, numbers, positions, cells, pbc, counts, packed_energies, packed_forces)
            merge_uncertainty_observations(prediction, observations, counts)
        else:
            add_observed_uncertainty(prediction, evaluator, [
                (numbers[start:end], positions[start:end], cell, pbc)
                for start, end, cell in zip(offsets[:-1], offsets[1:], cells)
            ])
    except ValueError as exc:
        raise ContractError(str(exc)) from exc
    return prediction


def check_evaluator(evaluator, atoms_list):
    """Check a sampler's evaluator, what a ``batched_calculator`` loader returns.

    An evaluator provides ``predict(atoms_list)`` (the layout :mod:`alframework.ml_interfaces.ensemble_inference`
    describes), ``evaluate(numbers, positions, cells, pbc, counts)`` on the packed tensors the ``torchsim`` and
    ``alchemi`` engines pass, returning energies ``[n_models, structures]`` and forces ``[n_models, atoms, 3]``, or
    both; each one it provides is checked, with any uncertainty of its own (``predict``'s ``"uncertainty"``, or
    ``last_uncertainty`` after ``evaluate``) and optional atom scores (``"atomic_uncertainty"`` or
    ``last_atomic_uncertainty``). For packed evaluators, also check the optional
    ``observe_uncertainty_batch`` or ``observe_uncertainty`` hook. Returns the per-structure signals acquisition
    would read.
    """
    members = getattr(evaluator, 'n_models', None)
    if isinstance(members, bool) or not isinstance(members, int) or members < 1:
        raise ContractError('An evaluator has a positive integer n_models.')
    methods = [name for name in ('predict', 'evaluate') if callable(getattr(evaluator, name, None))]
    if not methods:
        raise ContractError('An evaluator provides predict(atoms_list) or evaluate(numbers, positions, cells, pbc, '
                            'counts).')
    rows = None
    if 'predict' in methods:
        rows = _check_prediction(evaluator.predict(list(atoms_list)), members, atoms_list, 'predict()')
    if 'evaluate' in methods:
        packed = _check_prediction(_evaluate_packed(evaluator, list(atoms_list)), members, atoms_list, 'evaluate()')
        rows = packed if rows is None else rows
    return rows


def check_engine(engine, atoms_list, models, *, dt=0.5, temperature_K=300.0, steps=2, calculator_options=None):
    """Build a dynamics engine as the multi-trajectory sampler does, advance it ``steps`` steps, and check what the
    sampler calls. ``engine`` is a class, or an import path or built-in name as ``driver.engine`` takes. ``models``
    is what the sampler would give it: an evaluator, or a list of ASE calculators."""
    from alframework.samplers.engines.factory import make_engine
    name = engine if isinstance(engine, str) else f'{engine.__module__}.{engine.__qualname__}'
    rngs = [np.random.RandomState(index) for index in range(len(atoms_list))]
    options = dict({'well_params': None}, **(calculator_options or {}))
    built = make_engine(name, [atoms.copy() for atoms in atoms_list], models, dt, [temperature_K] * len(atoms_list),
                        rngs, options, device='cpu')
    missing = [method for method in ENGINE_METHODS if not callable(getattr(built, method, None))]
    if missing:
        raise ContractError(f'The engine has no {", ".join(missing)}.')
    indices = list(range(len(atoms_list)))
    failures = built.step(indices, steps)
    if not isinstance(failures, dict):
        raise ContractError('step(indices, n) returns a dict of {index: exception} for trajectories that failed.')
    if failures:
        raise ContractError(f'Trajectories failed while stepping: {failures}.')
    if [built.steps[index] for index in indices] != [steps] * len(indices):
        raise ContractError(f'After step(indices, {steps}), steps should be {steps} for each trajectory.')
    for index in indices:
        observation = built.observe(index)
        if not {'prediction', 'min_distance'} <= set(observation):
            raise ContractError('observe(index) returns {"prediction": ..., "min_distance": ...}.')
        if len(built.atoms(index)) != len(atoms_list[index]):
            raise ContractError('atoms(index) returns that trajectory\'s Atoms.')
    for index in indices:
        built.finish(index)
    return built
