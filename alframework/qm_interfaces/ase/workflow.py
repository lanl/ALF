"""Several ASE calculations on one fixed geometry, run in order in one QM worker.

Each stage is an independent single point: it starts from the molecule's original geometry and state, in its
own scratch directory, and nothing passes from one stage to the next. ``result_mode`` picks the label:

- ``last``: the final stage's results. Every stage must succeed.
- ``mean_if_agree``: identical stages repeated; their mean, if every pair agrees within ``agreement``.
"""
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import re

import numpy as np
from parsl import python_app

from alframework.tools.label_validation import validate_geometry, validate_label
from alframework.tools.molecules_class import MoleculesObject
from alframework.tools.task_timeline import timed
from .generic import ERROR_KEYS, _configuration, fresh_scratch_directory, label_one, molecule_directory

WORKFLOW_FIELDS = {'stages', 'result_mode', 'agreement', 'batching'}
STAGE_FIELDS = {'name', 'ASE_calculator', 'QM_run_command', 'input'}
RESULT_MODES = ('last', 'mean_if_agree')
# Stage names become directory names.
STAGE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
# Properties whose agreement mean_if_agree checks, in eV and eV/Angstrom.
AGREEMENT_PROPERTIES = ('energy', 'forces')


def _stage(stage, names, properties):
    """Check one stage; return its name, its settings and the calculator it resolves to."""
    if not isinstance(stage, dict):
        raise TypeError('Each ASE stage must be a dictionary.')
    unknown = set(stage) - STAGE_FIELDS
    if unknown:
        raise ValueError(f'Unknown ASE stage fields: {sorted(unknown)}.')
    name = stage.get('name')
    if not isinstance(name, str) or not STAGE_NAME.fullmatch(name):
        raise ValueError('Stage name must be a safe filename starting with a letter or digit.')
    if name in names:
        raise ValueError(f'Duplicate ASE stage name: {name!r}.')
    settings = deepcopy({key: value for key, value in stage.items() if key != 'name'})
    return name, settings, _configuration(settings, properties)


def _check_repeats(stages, identities, properties, tolerances):
    """mean_if_agree repeats one calculation, so the stages must be identical and both tolerances given."""
    if len(stages) < 2:
        raise ValueError('mean_if_agree requires at least two independent stages.')
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError('mean_if_agree requires identical calculator classes, commands, and options.')
    if properties is not None and not set(AGREEMENT_PROPERTIES) <= set(properties):
        raise ValueError('mean_if_agree requires requested energy and forces.')
    if not isinstance(tolerances, dict) or set(tolerances) != set(AGREEMENT_PROPERTIES):
        raise ValueError('agreement must provide energy (eV) and forces (eV/Angstrom) tolerances.')
    for key, value in tolerances.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'agreement {key} tolerance must be positive and finite.')


def _settings(config, properties=None):
    """Check a workflow configuration; return its stages as ``(name, settings)``, its mode and tolerances."""
    if not isinstance(config, dict):
        raise TypeError('ASE workflow configuration must be a dictionary.')
    unknown = set(config) - WORKFLOW_FIELDS
    if unknown:
        raise ValueError(f'Unknown ASE workflow fields: {sorted(unknown)}.')
    stages = config.get('stages')
    if not isinstance(stages, list) or not stages:
        raise ValueError('ASE workflow requires a nonempty stages list.')
    mode = config.get('result_mode', 'last')
    if mode not in RESULT_MODES:
        raise ValueError('result_mode must be last or mean_if_agree.')
    prepared, identities = [], []
    for stage in stages:
        name, settings, identity = _stage(stage, {name for name, _ in prepared}, properties)
        prepared.append((name, settings))
        identities.append(identity)
    tolerances = config.get('agreement')
    if mode == 'mean_if_agree':
        _check_repeats(prepared, identities, properties, tolerances)
    elif tolerances is not None:
        raise ValueError('agreement is only valid with mean_if_agree.')
    return prepared, mode, tolerances


def validate_ase_workflow_config(QM_config, master_config=None):
    """Validate every stage at coordinator startup without executing QM."""
    _settings(QM_config, (master_config or {}).get('properties_list'))


def _agreed_mean(completed, properties_list, tolerances, provenance):
    """The mean of the repeats' results, if every pair agrees within the tolerances; else ``ValueError``."""
    deviations = {key: 0.0 for key in AGREEMENT_PROPERTIES}
    for left, right in combinations(completed, 2):
        for key in properties_list:
            if np.shape(left.get_results()[key]) != np.shape(right.get_results()[key]):
                raise ValueError(f'ASE stages returned different shapes for {key}.')
        for key in deviations:
            with np.errstate(over='ignore', invalid='ignore'):
                difference = np.max(np.abs(np.asarray(left.get_results()[key]) - right.get_results()[key]))
            deviations[key] = max(deviations[key], float(difference))
            if not np.isfinite(difference) or difference >= tolerances[key]:
                provenance['max_deviation'] = deviations
                raise ValueError(f'ASE stages disagree on {key}: {difference} >= {tolerances[key]}.')
    provenance.update(agreement=deepcopy(tolerances), max_deviation=deviations)
    # Divide before summing, so large labels that agree can't overflow.
    return {key: sum(np.asarray(stage.get_results()[key]) / len(completed) for stage in completed)
            for key in properties_list}


def label_workflow(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Run the stages in this process and return the molecule, labeled or with the failing stage named.

    ``qm_workflow`` metadata records each stage's name, calculator, directory, convergence and electronic state,
    and for repeats the largest deviations. A failure sets ``qm_failed_stage`` to ``<stage>:<step>``.
    """
    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError('molecule_object must be a MoleculesObject.')
    molecule_object.set_converged_flag(False)
    records = []
    phase = 'configuration'
    mode = QM_config.get('result_mode', 'last') if isinstance(QM_config, dict) else None
    provenance = {'stages': records, 'result_mode': mode}
    try:
        stages, mode, tolerances = _settings(QM_config, properties_list)
        validate_geometry(molecule_object.get_atoms())
        directory = molecule_directory(molecule_object, QM_scratch_dir)
        phase = 'scratch'
        fresh_scratch_directory(directory)
        completed = []
        for name, settings in stages:
            phase = name
            stage_directory = str(Path(directory) / name)
            # Every stage starts from the original geometry and metadata.
            stage = MoleculesObject(molecule_object.get_atoms().copy(), molecule_object.get_moleculeid())
            stage.metadata = deepcopy(molecule_object.metadata)
            label_one(stage, settings, QM_scratch_dir, properties_list, directory=stage_directory)
            record = {'name': name, 'calculator': settings['ASE_calculator'], 'directory': stage_directory,
                      'converged': bool(stage.check_convergence())}
            records.append(record)
            if not stage.check_convergence():
                record.update({key: stage.metadata[key] for key in ERROR_KEYS if key in stage.metadata})
                molecule_object.update_metadata({key: stage.metadata[key] for key in ('qm_error', 'qm_error_type')})
                molecule_object.metadata['qm_failed_stage'] = name + ':' + stage.metadata['qm_failed_stage']
                molecule_object.metadata['qm_workflow'] = provenance
                return molecule_object
            record['state'] = {key: deepcopy(stage.metadata[key]) for key in ('qm_backend', 'charge', 'mult')
                               if key in stage.metadata}
            completed.append(stage)
        phase = 'agreement'
        if mode == 'mean_if_agree':
            results = _agreed_mean(completed, properties_list, tolerances, provenance)
        else:
            results = {key: deepcopy(completed[-1].get_results()[key]) for key in properties_list}
        phase = 'validation'
        final = MoleculesObject(molecule_object.get_atoms().copy(), molecule_object.get_moleculeid())
        final.store_results(results)
        validate_label(final, properties_list)
    except Exception as error:
        molecule_object.update_metadata({'qm_error': f'{type(error).__name__}: {error}',
                                         'qm_error_type': type(error).__name__, 'qm_failed_stage': phase,
                                         'qm_workflow': provenance})
        return molecule_object
    molecule_object.store_results(results)
    molecule_object.update_metadata(deepcopy(completed[-1].metadata))
    molecule_object.metadata['qm_workflow'] = provenance
    molecule_object.set_converged_flag(True)
    for key in ERROR_KEYS:
        molecule_object.metadata.pop(key, None)
    return molecule_object


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def ase_workflow_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Label a fixed geometry through ordered upstream ASE calculations."""
    return label_workflow(molecule_object, QM_config, QM_scratch_dir, properties_list)


# The coordinator calls this at startup. Without Parsl's decorator (as in the docs build), the task is the function.
getattr(ase_workflow_task, 'func', ase_workflow_task).validate_config = validate_ase_workflow_config
