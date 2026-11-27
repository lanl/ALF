"""data_by_reference labels: QM workers write their labels as parts and return receipts; the coordinator adopts the
receipts and merges the parts before publishing. Snapshots must hold exactly what the coordinator's own parts would."""
import json

import h5py
import numpy as np
import pytest
from ase import Atoms

from alframework.tools.label_parts import (
    INCOMING, MANIFEST, MERGED_FROM, HDF5HeldLabels, LabelReceipt, incoming_directory, part_directory,
    remove_orphan_worker_parts, write_label_part)
from alframework.tools.molecules import MoleculesObject
from tests.helpers.molecules import WATER
from tests.test_restarts import PROPERTIES, labeled

SCHEMA = json.dumps(PROPERTIES, sort_keys=True)


def _labels(count, prefix='mol'):
    return [labeled(f'{prefix}-{index:04d}', energy=-1.0 - index) for index in range(count)]


def _write(results, snapshot):
    return write_label_part(results, incoming_directory(snapshot), PROPERTIES, SCHEMA)


def _contents(snapshot):
    with h5py.File(snapshot) as handle:
        return {formula: {name: dataset[()] for name, dataset in group.items()} for formula, group in handle.items()}


def _assert_same(first, second):
    assert set(first) == set(second)
    for formula, columns in first.items():
        assert set(columns) == set(second[formula])
        for name, value in columns.items():
            np.testing.assert_array_equal(value, second[formula][name])


def test_a_worker_part_screens_labels_and_records_why_each_was_rejected(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    good = _labels(2)
    unconverged = labeled('mol-unconverged')
    unconverged.set_converged_flag(False)
    unconverged.update_metadata({'labeling_error': 'SCF did not converge'})
    invalid = MoleculesObject(WATER.copy(), 'mol-invalid')
    invalid.store_results({'energy': -1.0})  # no forces
    invalid.set_converged_flag(True)
    periodic = labeled('mol-periodic')
    periodic.get_atoms().set_cell([10, 10, 10])
    periodic.get_atoms().set_pbc(True)
    receipt = _write(good + [unconverged, invalid, periodic], snapshot)
    assert receipt.formulas == {'H02_O01': {'ids': ['mol-0000', 'mol-0001'], 'periodic': False}}
    assert receipt.rejected['mol-unconverged'] == 'SCF did not converge'
    assert 'forces' in receipt.rejected['mol-invalid']
    assert 'cannot mix periodic and nonperiodic' in receipt.rejected['mol-periodic']
    assert len(receipt) == 5 and sorted(receipt.ids()) == sorted(m.get_moleculeid() for m in
                                                                 good + [unconverged, invalid, periodic])
    with h5py.File(receipt.path) as handle:
        assert list(handle['H02_O01']['_id'].asstr()[:]) == ['mol-0000', 'mol-0001']
        assert sorted(json.loads(handle.attrs['alf_rejected_label_ids'])) == sorted(receipt.rejected)
    nothing = labeled('mol-x')
    nothing.set_converged_flag(False)
    assert _write([nothing], snapshot).path is None, 'no part without a converged label'


@pytest.mark.parametrize('merge_workers', [1, 2])
def test_merged_worker_parts_publish_the_same_snapshot_as_the_coordinators_parts(tmp_path, merge_workers):
    samples = _labels(7)
    samples[2].update_metadata({'charge': -1})  # charge and multiplicity in some parts only
    rejected = labeled('mol-rejected')
    rejected.set_converged_flag(False)

    control = tmp_path / 'control' / 'data-0000.h5'
    held = HDF5HeldLabels(control, PROPERTIES, batch_size=3)
    for chunk in (samples[:2], samples[2:5] + [rejected], samples[5:]):
        held.hold([m.copy() if hasattr(m, 'copy') else m for m in chunk])
    assert held.commit(control, PROPERTIES, {}) == 7

    snapshot = tmp_path / 'by_reference' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=3, merge_workers=merge_workers)
    receipts = [_write(chunk, snapshot) for chunk in (samples[:2], samples[2:5] + [rejected], samples[5:])]
    held.adopt(receipts[:2])
    held.adopt(receipts[2:])
    assert (held.successful(), held.converged()) == (8, 7)
    assert held.molecule_ids() == {m.get_moleculeid() for m in samples} | {'mol-rejected'}
    assert held.commit(snapshot, PROPERTIES, {}) == 7
    _assert_same(_contents(snapshot), _contents(control))
    assert not list(incoming_directory(snapshot).glob('task-*.h5')), 'worker parts go once merged and published'
    merged = sorted(part_directory(snapshot).glob('part-*.h5'))
    assert len(merged) <= 3, 'merged into parts of about batch_size labels'


def test_a_restart_counts_adopted_receipts_once_before_and_after_the_merge(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    failed = labeled('mol-failed')
    failed.set_converged_flag(False)
    held.adopt([_write(_labels(3), snapshot), _write([failed], snapshot)])
    counts = (held.successful(), held.converged(), held.molecule_ids())

    restarted = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    assert (restarted.successful(), restarted.converged(), restarted.molecule_ids()) == counts
    restarted.merge()  # a crash after merging, before publishing
    again = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=2)
    assert (again.successful(), again.converged(), again.molecule_ids()) == counts
    assert again.merge_jobs() == [], 'merged receipts are not merged again'
    with h5py.File(sorted(part_directory(snapshot).glob('part-*.h5'))[0]) as handle:
        assert json.loads(handle.attrs[MERGED_FROM])
    assert again.commit(snapshot, PROPERTIES, {}) == 3


def test_a_manifest_line_cut_short_by_a_crash_was_never_adopted(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4)
    held.adopt([_write(_labels(2), snapshot)])
    with open(part_directory(snapshot) / MANIFEST, 'a') as stream:
        stream.write('{"path": "cut sh')
    restarted = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4)
    assert restarted.converged() == 2
    restarted.adopt([_write(_labels(1, 'more'), snapshot)])
    assert HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4).converged() == 3
    assert restarted.commit(snapshot, PROPERTIES, {}) == 3


@pytest.mark.parametrize('grouped', [False, True])
def test_a_formula_that_changes_periodicity_across_parts_is_rejected_from_the_later_part(tmp_path, grouped):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=8)
    periodic = labeled('mol-periodic')
    periodic.get_atoms().set_cell([10, 10, 10])
    periodic.get_atoms().set_pbc(True)
    receipts = [_write(_labels(2), snapshot), _write([periodic], snapshot)]
    if grouped:
        held.adopt(receipts)
    else:
        for receipt in receipts:
            held.adopt([receipt])
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=8)
    assert (held.successful(), held.converged()) == (3, 2)
    assert held.commit(snapshot, PROPERTIES, {}) == 2
    with h5py.File(snapshot) as handle:
        assert 'cell' not in handle['H02_O01']


def test_merging_worker_labels_checks_their_original_conversion(tmp_path):
    snapshot = tmp_path / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=8)
    outdated = dict(PROPERTIES, energy=['energy', 'system', 2.0])
    receipt = write_label_part(_labels(1), incoming_directory(snapshot), outdated,
                               json.dumps(outdated, sort_keys=True))
    held.adopt([receipt])
    with pytest.raises(ValueError, match='label properties or units differ'):
        held.commit(snapshot, PROPERTIES, {})
    assert not snapshot.exists() and list(incoming_directory(snapshot).glob('*.h5'))


def test_worker_parts_no_generation_adopted_are_removed_at_startup(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4)
    adopted = _write(_labels(1), snapshot)
    held.adopt([adopted])
    orphan = _write(_labels(1, 'orphan'), snapshot)
    assert remove_orphan_worker_parts(HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4)) == 1
    assert not __import__('os').path.exists(orphan.path) and __import__('os').path.exists(adopted.path)


def test_keep_files_keeps_the_worker_parts(tmp_path):
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, keep_worker_parts=True)
    receipt = _write(_labels(2), snapshot)
    held.adopt([receipt])
    held.commit(snapshot, PROPERTIES, {})
    assert __import__('os').path.exists(receipt.path)


def test_a_receipt_travels_as_a_small_record():
    import pickle
    receipt = LabelReceipt('/x/task-1.h5', {'H2_O': {'ids': ['a', 'b'], 'periodic': False}}, {'c': 'reason'})
    copy = pickle.loads(pickle.dumps(receipt))
    assert copy.as_record() == receipt.as_record() and len(copy) == 3
    assert LabelReceipt.from_record(receipt.as_record()).ids() == ['a', 'b', 'c']


def test_labels_by_reference_needs_native_qm_batching():
    from tests.test_labeling_coordinator import native_qm, single_qm
    from alframework.tools.labeling_coordinator import LabelingCoordinator
    handoff = dict(directory='/tmp/x', properties=PROPERTIES, schema=SCHEMA)
    with pytest.raises(ValueError, match='data_by_reference labels requires native QM batching'):
        LabelingCoordinator(single_qm, 'test.single_qm', {}, label_handoff=handoff)
    coordinator = LabelingCoordinator(native_qm, 'test.native_qm', {'batching': {'mode': 'native', 'batch_size': 2}},
                                      label_handoff=handoff)
    assert LabelReceipt in coordinator.queue.output_types


def test_the_coordinator_stores_the_same_labels_with_labels_by_reference(tmp_path, monkeypatch):
    """The same finite bootstrap, labeled the usual way and by reference, stores the same labels in its first
    shard; by reference, the QM workers wrote them and the coordinator only merged them."""
    from alframework.tools import config as workflow_config, sampled_frames
    from tests.helpers.driver import run_alf_driver, stored_ids
    from tests.test_labeling_coordinator import native_qm
    from tests.test_restarts import LOOP, restart_tasks

    first = {}
    for arm in ('coordinator', 'by_reference'):
        run_dir = tmp_path / arm
        run_dir.mkdir()
        extra = {} if arm == 'coordinator' else {'data_by_reference': {'labels': True, 'merge_workers': 2}}
        with monkeypatch.context() as patch:
            # The by-reference QM wrapper looks its labeler up by name, as the driver's stand-ins are.
            patch.setattr(sampled_frames, 'load_module_from_string',
                          lambda path: workflow_config.load_module_from_string(path))
            tasks = restart_tasks(run_dir, [])
            tasks['qm'] = native_qm
            run = run_alf_driver(run_dir, patch, **tasks, **LOOP, finite_bootstrap=True, label_storage_batch_size=2,
                                 qm_config={'batching': {'mode': 'native', 'batch_size': 2,
                                                         'partial_policy': 'timeout', 'max_wait_seconds': 0.1}},
                                 **extra)
        assert run.status['lifetime_stored_labels'] > 0
        with h5py.File(run_dir / 'h5' / 'data-0000.h5') as handle:
            first[arm] = sorted(i.decode() for group in handle.values() for i in group['_id'][()])
        if arm == 'by_reference':
            incoming = run_dir / 'h5' / '.alf-label-parts' / INCOMING
            assert incoming.is_dir(), 'QM workers wrote parts'
            ids = stored_ids(run_dir / 'h5')
            assert len(ids) == len(set(ids)), 'no label was stored twice'
    assert first['by_reference'] == first['coordinator']


def test_a_real_parsl_worker_writes_the_part_and_returns_its_receipt(tmp_path, monkeypatch):
    """On a worker process, as at scale: the receipt crosses Parsl's serialization and the labeler is imported
    by its path in the worker."""
    import parsl
    from parsl import python_app
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider
    from alframework.tools.sampled_frames import label_sampled_frames

    import os
    import sys
    root = str(__import__('pathlib').Path(__file__).resolve().parents[1])
    monkeypatch.setenv('PYTHONPATH', root + os.pathsep + os.environ.get('PYTHONPATH', ''))
    # Parsl starts its interchange and worker pool from PATH: this interpreter's, wherever it is installed.
    monkeypatch.setenv('PATH', os.path.dirname(sys.executable) + os.pathsep + os.environ.get('PATH', ''))
    config = Config(run_dir=str(tmp_path / 'runinfo'), executors=[HighThroughputExecutor(
        label='alf_QM_executor', address='127.0.0.1', max_workers_per_node=1, cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, min_blocks=0, max_blocks=1))])
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    handoff = dict(directory=str(incoming_directory(snapshot)), properties=PROPERTIES, schema=SCHEMA)
    parsl.load(config)
    try:
        task = python_app(executors=['alf_QM_executor'])(label_sampled_frames)
        structures = [MoleculesObject(WATER.copy(), f'mol-{index:04d}') for index in range(3)]
        receipt = task(structures, 'tests.test_labeling_coordinator.native_qm', {'QM_config': {}}, handoff).result(
            timeout=120)
    finally:
        parsl.dfk().cleanup()
    assert isinstance(receipt, LabelReceipt) and len(receipt) == 3
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=8)
    held.adopt([receipt])
    assert held.commit(snapshot, PROPERTIES, {}) == 3


def _mixed(count):
    """Labels of three formulas, one with a charge, so a snapshot has several groups and state columns."""
    labels = []
    for index in range(count):
        formula = ('H2O', 'NH3', 'CH4')[index % 3]
        atoms = Atoms(formula, positions=np.arange(3.0 * len(Atoms(formula))).reshape(-1, 3))
        molecule = MoleculesObject(atoms, f'mol-{index:04d}')
        molecule.store_results({'energy': -1.0 - index, 'forces': np.zeros((len(atoms), 3))})
        molecule.set_converged_flag(True)
        if index == 4:
            molecule.update_metadata({'charge': -1})
        labels.append(molecule)
    return labels


def _virtual_layout_columns(group, segments, write_state):
    """The virtual columns as h5py's VirtualLayout writes them, for comparison with the low-level writer."""
    from alframework.tools.label_parts import STATE_DEFAULTS
    keys = set(segments[0][2]) - STATE_DEFAULTS.keys()
    frames = sum(size for _, size, _ in segments)
    if write_state:
        keys.update(STATE_DEFAULTS)
    for name in sorted(keys):
        shape = next(iter({columns[name][1][1:] for _, _, columns in segments if name in columns}), ())
        dtypes = [columns[name][0] for _, _, columns in segments if name in columns]
        dtype = np.result_type(*dtypes) if dtypes else np.int64
        if not dtypes:
            group.create_dataset(name, shape=(frames,), dtype=dtype, fillvalue=STATE_DEFAULTS[name])
            continue
        layout = h5py.VirtualLayout(shape=(frames, *shape), dtype=dtype)
        offset = 0
        for source, size, columns in segments:
            if name in columns:
                layout[offset:offset + size] = h5py.VirtualSource(source, f'{group.name}/{name}',
                                                                  shape=columns[name][1])
            offset += size
        group.create_virtual_dataset(name, layout, fillvalue=STATE_DEFAULTS.get(name))


def _described(snapshot):
    """Each dataset's type, shape, fill value, mappings and values."""
    described = {}
    with h5py.File(snapshot) as handle:
        for formula, group in handle.items():
            for name, dataset in group.items():
                mappings = [(m.file_name, m.dset_name, m.vspace.get_select_type(), m.vspace.get_select_bounds(),
                             m.src_space.get_select_type(), m.src_space.shape)
                            for m in dataset.virtual_sources()] if dataset.is_virtual else None
                described[f'{formula}/{name}'] = (dataset.dtype.str, dataset.shape, dataset.fillvalue, mappings,
                                                  dataset[()].tobytes())
    return described


def test_a_snapshot_maps_its_parts_as_hdf5s_virtual_layout_would(tmp_path, monkeypatch):
    from alframework.tools import label_parts
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, keep_worker_parts=True)
    labels = _mixed(11)
    held.adopt([_write(labels[start:start + 3], snapshot) for start in range(0, 11, 3)])
    held.commit(snapshot, PROPERTIES, {})
    parts = sorted(part_directory(snapshot).glob('part-*.h5'))
    assert len(parts) > 1, 'the test needs groups mapped from several parts'

    reference = snapshot.with_name('reference.h5')
    monkeypatch.setattr(label_parts, '_write_virtual_columns', _virtual_layout_columns)
    label_parts.publish_snapshot(reference, parts, {})
    ours, theirs = _described(snapshot), _described(reference)
    assert ours.keys() == theirs.keys() and any(value[3] for value in ours.values())
    for key, value in ours.items():
        assert value == theirs[key], key


def test_publishing_after_a_merge_reuses_the_merged_parts_layouts(tmp_path, monkeypatch):
    from alframework.tools import label_parts
    snapshot = tmp_path / 'h5' / 'data-0000.h5'
    held = HDF5HeldLabels(snapshot, PROPERTIES, batch_size=4, merge_workers=1)
    labels = _mixed(9)
    held.adopt([_write(labels[start:start + 3], snapshot) for start in range(0, 9, 3)])
    held.merge()
    scanned = []
    layout = label_parts.part_layout
    monkeypatch.setattr(label_parts, 'part_layout', lambda part: scanned.append(part) or layout(part))
    assert held.commit(snapshot, PROPERTIES, {}) == 9
    # At scale, rescanning every merged part's groups took most of the publish time.
    assert not scanned, 'the snapshot was built from the layouts the merge recorded'
