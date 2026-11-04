"""Original-example validation must preserve scientific settings and label lineage."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/simple_water_campaign'))
import campaign_support as support
import compatibility
from tests.test_water_campaign import members, write_h5


@pytest.mark.parametrize('family', compatibility.FAMILIES)
def test_compatibility_copies_original_methods_and_training_settings(tmp_path, family):
    run = compatibility.prepare(tmp_path/family, family, support.darwin_site())
    example = support.SOURCE/'examples'/family
    original = support.read_json(example/'master_config.json')
    master = support.read_json(run/'master_config.json')
    spec = support.read_json(run/'run.json')
    assert spec['compatibility']['tasks'] == {k: original[k] for k in ('builder_task','sampler_task','QM_task','ML_task')}
    assert master['properties_list'] == original['properties_list']
    sampler = support.read_json(run/'sampler_config.json')
    old_sampler = support.read_json(example/original['sampler_config_path'])
    assert {k:v for k,v in sampler.items() if k!='maxt'} == {k:v for k,v in old_sampler.items() if k!='maxt'}
    ml = support.read_json(run/'ml_config.json')
    old_ml = support.read_json(example/original['ML_config_path'])
    assert {k:v for k,v in ml.items() if k!='n_models'} == {k:v for k,v in old_ml.items() if k!='n_models'}
    qm = support.read_json(run/'qm_config.json')
    old_qm = support.read_json(example/original['QM_config_path'])
    if 'orcasimpleinput' in old_qm:
        assert qm['orcasimpleinput'] == old_qm['orcasimpleinput']
    assert all('batching' not in c for c in (sampler, qm))
    compatibility.prepare(run, family, support.darwin_site())
    ml['learning_rate'] = .99
    support.atomic_json(run/'ml_config.json', ml)
    with pytest.raises(ValueError, match='Immutable'):
        compatibility.prepare(run, family, support.darwin_site())


def test_original_task_gets_its_native_arguments_and_model_generation(tmp_path, monkeypatch):
    run = compatibility.prepare(tmp_path, 'fine_tuning', support.darwin_site())
    monkeypatch.setenv('ALF_CAMPAIGN_RUN', str(run))
    calls = []
    def native(ML_config, current_model_id, h5_dir):
        calls.append((ML_config, current_model_id, h5_dir))
        return 'unchanged result'
    monkeypatch.setattr(compatibility.importlib, 'import_module', lambda name: SimpleNamespace(
        fine_tune_HIPPYNN_ensemble_task=SimpleNamespace(func=native)))
    result = compatibility.original('ML_task', dict(ML_config={'learning_rate':.0001},
        current_model_id=7, h5_dir='case/h5store', unrelated='not a task argument'))
    assert result == 'unchanged result'
    assert calls == [({'learning_rate':.0001}, 7, 'case/h5store')]
    assert support.events(run)[0]['task'] == 'fine_tune_ml_task.fine_tune_HIPPYNN_ensemble_task'


def test_seeded_example_gets_labels_without_inheriting_models_or_status(tmp_path):
    seed = tmp_path/'seed'
    write_h5(seed/'h5store/data-0000.h5', ['mol-boot-0000000000'])
    model = seed/'models/model-0000/member'
    model.parent.mkdir(parents=True)
    model.write_text('initial weights')
    files = {str(p.relative_to(seed)):support.fingerprint(p) for p in seed.rglob('*') if p.is_file()}
    support.atomic_json(seed/'seed.json', dict(ids=['mol-boot-0000000000'], files_sha256=files))
    run = tmp_path/'case'
    compatibility.import_water_seed(seed, run, 'seeded_active_learning')
    assert not (run/'models').exists() and not (run/'status.json').exists()
    assert (seed/'h5store/data-0000.h5').stat().st_ino != (run/'h5store/data-0000.h5').stat().st_ino
    assert set(support.read_json(run/'seed.json')['files_sha256']) == {'h5store/data-0000.h5'}
    with pytest.raises(ValueError, match='already exists'):
        compatibility.import_water_seed(seed, run, 'seeded_active_learning')


def test_pbe_campaign_seed_cannot_be_mixed_into_original_example_labels(tmp_path):
    bootstrap = compatibility.prepare(tmp_path/'bootstrap', 'simple_water', support.darwin_site())
    run = compatibility.prepare(tmp_path/'case', 'fine_tuning', support.darwin_site())
    support.atomic_json(bootstrap/'result.json', dict(status='passed', accepted=dict(training=dict(members=members()))))
    compatibility.verify_water_seed(tmp_path/'seed', run)
    qm = support.read_json(bootstrap/'qm_config.json')
    qm['orcasimpleinput'] = 'EnGrad PBE def2-SVP def2/J RI TightSCF Float Hirshfeld'
    support.atomic_json(bootstrap/'qm_config.json', qm)
    with pytest.raises(ValueError, match='QM method differs'):
        compatibility.verify_water_seed(tmp_path/'seed', run)
