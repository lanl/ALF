"""Several ASE calculations on one fixed geometry, run in order in one QM worker.

Each stage is a single point on the molecule's original geometry and state, with a fresh calculator in its own
scratch directory. Nothing passes between stages except files a stage declares in ``inputs_from_stages``, such
as orbitals or a charge density to start from. ``result_mode`` picks the label:

- ``last``: the final stage's results. Every stage must succeed; earlier stages may compute fewer properties.
- ``mean_if_agree``: identical stages repeated; their mean, if every pair agrees within ``agreement``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import re

import ase
import numpy as np
from parsl import python_app

from alframework.tools.label_validation import validate_geometry, validate_label
from alframework.tools.molecules import MoleculesObject
from alframework.tools.task_timeline import timed
from alframework.tools.label_sources import LABEL_SOURCE
from .generic import (ERROR_KEYS, _calculator_options, _configuration, _remove_scratch, fresh_scratch_directory,
                      label_one, molecule_directory, remove_scratch_directory)

WORKFLOW_FIELDS = {'stages', 'result_mode', 'agreement', 'batching', 'remove_scratch_on_success'}
STAGE_CONTROLS = {'name', 'properties', 'inputs_from_stages'}
STAGE_FIELDS = STAGE_CONTROLS | {'ASE_calculator', 'QM_run_command', 'input', 'initial_magnetic_moments'}
RESULT_MODES = ('last', 'mean_if_agree')
# Stage names become directory names.
STAGE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
# Properties whose agreement mean_if_agree checks, in eV and eV/Angstrom.
AGREEMENT_PROPERTIES = ('energy', 'forces')


@dataclass
class _Stage:
    """One checked stage: its calculator settings, the properties it computes, and the files it reads."""
    name: str
    settings: dict
    properties: dict | None
    inputs: list
    # Parallel to ``inputs``: whether this is the file's last use, so it can be moved rather than copied.
    moves: list = field(default_factory=list)


def _stage_properties(stage, properties, mode, final):
    """The properties a stage computes: all of them, or the subset a preparatory stage names.

    Only stages before the last in ``last`` mode may compute fewer, since only the last stage gives the label.
    """
    if 'properties' not in stage:
        return properties
    if mode != 'last' or final:
        raise ValueError('Only preparatory stages in result_mode last can choose properties.')
    requested = stage['properties']
    if (not isinstance(requested, list) or not requested
            or any(not isinstance(key, str) or not key for key in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError('Stage properties must be a nonempty list of distinct property names.')
    if properties is None:
        return None
    unknown = set(requested) - properties.keys()
    if unknown:
        raise ValueError(f'Stage properties must be in properties_list. Unknown properties: {sorted(unknown)}.')
    return {key: properties[key] for key in requested}


def _file_name(value):
    """Whether ``value`` is a bare file name, so a handoff can't read or write outside its stage directory."""
    return (isinstance(value, str) and value not in ('', '.', '..')
            and not any(character in value for character in ('/', '\\', '\0')))


def _stage_inputs(stage, names, mode):
    """Check a stage's ``inputs_from_stages``: each names an earlier stage, a file there, and a new name here."""
    if 'inputs_from_stages' not in stage:
        return []
    if mode != 'last':
        raise ValueError('inputs_from_stages requires result_mode last.')
    inputs = stage['inputs_from_stages']
    if not isinstance(inputs, list):
        raise ValueError('inputs_from_stages must be a list.')
    destinations = set()
    for entry in inputs:
        if not isinstance(entry, dict) or entry.keys() != {'stage', 'source', 'destination'}:
            raise ValueError('Each stage input must specify stage, source, and destination.')
        if not isinstance(entry['stage'], str) or entry['stage'] not in names:
            raise ValueError(f"Stage input must refer to an earlier stage: {entry['stage']!r}.")
        if not _file_name(entry['source']) or not _file_name(entry['destination']):
            raise ValueError('Stage input source and destination must be filenames without directories.')
        destination = entry['destination']
        if destination in destinations:
            raise ValueError(f'Duplicate stage input destination: {destination!r}.')
        destinations.add(destination)
    return deepcopy(inputs)


def _stage(stage, names, properties, mode, final):
    """Check one stage and resolve its calculator identity."""
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
    selected = _stage_properties(stage, properties, mode, final)
    inputs = _stage_inputs(stage, names, mode)
    settings = deepcopy({key: value for key, value in stage.items() if key not in STAGE_CONTROLS})
    return _Stage(name, settings, selected, inputs), _configuration(settings, selected)


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
    """Check a workflow configuration and return its prepared stages, mode and tolerances."""
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
    names = set()
    for index, stage in enumerate(stages):
        spec, identity = _stage(stage, names, properties, mode, final=index == len(stages) - 1)
        prepared.append(spec)
        identities.append(identity)
        names.add(spec.name)
    tolerances = config.get('agreement')
    if mode == 'mean_if_agree':
        _check_repeats(prepared, identities, properties, tolerances)
    elif tolerances is not None:
        raise ValueError('agreement is only valid with mean_if_agree.')
    _remove_scratch(config)
    _mark_last_uses(prepared)
    return prepared, mode, tolerances


def _mark_last_uses(stages):
    """Mark each declared input that is the last to read its source file, so ALF can move that file."""
    last = {}
    for stage_index, spec in enumerate(stages):
        for entry_index, entry in enumerate(spec.inputs):
            last[(entry['stage'], entry['source'])] = (stage_index, entry_index)
    for stage_index, spec in enumerate(stages):
        spec.moves = [last[(entry['stage'], entry['source'])] == (stage_index, entry_index)
                      for entry_index, entry in enumerate(spec.inputs)]


def validate_ase_workflow_config(QM_config, master_config=None):
    """Validate every stage at coordinator startup without executing QM."""
    _settings(QM_config, (master_config or {}).get('properties_list'))


def _agreed_mean(completed, properties_list, tolerances, workflow_record):
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
                workflow_record['max_deviation'] = deviations
                raise ValueError(f'ASE stages disagree on {key}: {difference} >= {tolerances[key]}.')
    workflow_record.update(agreement=deepcopy(tolerances), max_deviation=deviations)
    # Divide before summing, so large labels that agree can't overflow.
    return {key: sum(np.asarray(stage.get_results()[key]) / len(completed) for stage in completed)
            for key in properties_list}


def _workflow_label_source(stages, mode, tolerances):
    """The label source of a workflow label: the mode, and each stage's calculator, settings and inputs."""
    records = []
    for spec in stages:
        record = {'name': spec.name, 'calculator': spec.settings['ASE_calculator'],
                  'input': deepcopy(_calculator_options(spec.settings))}
        if 'initial_magnetic_moments' in spec.settings:
            record['initial_magnetic_moments'] = deepcopy(spec.settings['initial_magnetic_moments'])
        if spec.properties is not None:
            record['properties'] = list(spec.properties)
        if spec.inputs:
            record['inputs_from_stages'] = deepcopy(spec.inputs)
        records.append(record)
    source = {'labeler': 'ase_workflow_task', 'ase_version': ase.__version__, 'result_mode': mode,
              'stages': records}
    if tolerances is not None:
        source['agreement'] = deepcopy(tolerances)
    return source


def label_workflow(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Run the stages in this process and return the molecule, labeled or with the failing stage named.

    A good label records the workflow in its ``label_source``, the same for every label, and for repeats its
    largest deviations in ``qm_agreement_max_deviation``. A failed label instead records ``qm_workflow``: each
    stage that ran, with its directory, convergence and electronic state or error, so the failure can be
    inspected; ``qm_failed_stage`` is ``<stage>:<step>``.
    """
    if not isinstance(molecule_object, MoleculesObject):
        raise TypeError('molecule_object must be a MoleculesObject.')
    molecule_object.set_converged_flag(False)
    records = []
    phase = 'configuration'
    mode = QM_config.get('result_mode', 'last') if isinstance(QM_config, dict) else None
    workflow_record = {'stages': records, 'result_mode': mode}
    try:
        stages, mode, tolerances = _settings(QM_config, properties_list)
        validate_geometry(molecule_object.get_atoms())
        directory = molecule_directory(molecule_object, QM_scratch_dir)
        phase = 'scratch'
        fresh_scratch_directory(directory)
        completed = []
        for spec in stages:
            phase = spec.name
            stage_directory = str(Path(directory) / spec.name)
            input_files = [(Path(directory) / entry['stage'] / entry['source'], entry['destination'], move)
                           for entry, move in zip(spec.inputs, spec.moves)]
            # Every stage starts from the original geometry and metadata.
            stage = MoleculesObject(molecule_object.get_atoms().copy(), molecule_object.get_moleculeid())
            stage.metadata = deepcopy(molecule_object.metadata)
            label_one(stage, spec.settings, QM_scratch_dir, spec.properties, directory=stage_directory,
                      input_files=input_files)
            record = {'name': spec.name, 'calculator': spec.settings['ASE_calculator'], 'directory': stage_directory,
                      'converged': bool(stage.check_convergence())}
            if spec.inputs:
                record['inputs_from_stages'] = spec.inputs
            records.append(record)
            if not stage.check_convergence():
                record.update({key: stage.metadata[key] for key in ERROR_KEYS if key in stage.metadata})
                molecule_object.update_metadata({key: stage.metadata[key] for key in ('qm_error', 'qm_error_type')})
                molecule_object.metadata['qm_failed_stage'] = spec.name + ':' + stage.metadata['qm_failed_stage']
                molecule_object.metadata['qm_workflow'] = workflow_record
                return molecule_object
            record['state'] = {key: deepcopy(stage.metadata[key]) for key in ('qm_backend', 'charge', 'mult')
                               if key in stage.metadata}
            completed.append(stage)
        phase = 'agreement'
        if mode == 'mean_if_agree':
            results = _agreed_mean(completed, properties_list, tolerances, workflow_record)
        else:
            results = {key: deepcopy(completed[-1].get_results()[key]) for key in properties_list}
        phase = 'validation'
        final = MoleculesObject(molecule_object.get_atoms().copy(), molecule_object.get_moleculeid())
        final.store_results(results)
        validate_label(final, properties_list)
    except Exception as error:
        molecule_object.update_metadata({'qm_error': f'{type(error).__name__}: {error}',
                                         'qm_error_type': type(error).__name__, 'qm_failed_stage': phase,
                                         'qm_workflow': workflow_record})
        return molecule_object
    molecule_object.store_results(results)
    molecule_object.update_metadata(deepcopy(completed[-1].metadata))
    # The stages' settings are in the label source, and their directories follow from the stage names, so a
    # good label carries only what differs from label to label.
    molecule_object.metadata.pop('qm_workflow', None)
    if mode == 'mean_if_agree':
        molecule_object.metadata['qm_agreement_max_deviation'] = workflow_record['max_deviation']
    molecule_object.metadata[LABEL_SOURCE] = _workflow_label_source(stages, mode, tolerances)
    molecule_object.set_converged_flag(True)
    for key in ERROR_KEYS:
        molecule_object.metadata.pop(key, None)
    if _remove_scratch(QM_config):
        remove_scratch_directory(directory)
    return molecule_object


@python_app(executors=['alf_QM_executor'])
@timed('QM')
def ase_workflow_task(molecule_object, QM_config, QM_scratch_dir, properties_list):
    """Label a fixed geometry through ordered upstream ASE calculations."""
    return label_workflow(molecule_object, QM_config, QM_scratch_dir, properties_list)


# The coordinator calls this at startup. Without Parsl's decorator (as in the docs build), the task is the function.
getattr(ase_workflow_task, 'func', ase_workflow_task).validate_config = validate_ase_workflow_config
