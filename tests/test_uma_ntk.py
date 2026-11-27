"""UMA checkpoint export and NTK provenance, without requiring foundation weights in CI."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from alframework.ml_interfaces.fine_tuning.uma_lora import (export_lora_checkpoint, install_lora,
                                                            restore_ema_lora)
from alframework.ml_interfaces.ntk.uma import _member_artifacts, load_uma_ntk_estimator


def container(**children):
    module = torch.nn.Module()
    for name, child in children.items():
        setattr(module, name, child)
    return module


def radial_model():
    radial = container(net=torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU(), torch.nn.Identity(),
                                               torch.nn.Linear(4, 4)))
    block = container(edge_wise=container(so2_conv_1=container(rad_func=radial)))
    backbone = container(blocks=torch.nn.ModuleList([block]), condition=torch.nn.Linear(2, 4))
    backbone.regress_config = SimpleNamespace(direct_forces=False)
    return container(backbone=backbone, output_heads=torch.nn.Linear(4, 1))


def test_uma_export_merges_regular_and_ema_and_restores_exact_ema_factors(tmp_path):
    model = radial_model()
    initial = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    targets = install_lora(model, rank=2, alpha=1.0)
    assert all(torch.equal(module.weight, initial[f'{name}.weight']) for name, module in targets.items())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {f'{name}.parametrizations.weight.0.{factor}' for name in targets for factor in ('a', 'b')} | {
        'output_heads.weight', 'output_heads.bias'}
    ema = torch.optim.swa_utils.AveragedModel(model)
    for module in targets.values():
        torch.nn.init.constant_(module.parametrizations.weight[0].b, 0.2)
    for module in ema.module.modules():
        if hasattr(module, 'parametrizations'):
            torch.nn.init.constant_(module.parametrizations.weight[0].b, 0.4)
    original = tmp_path / 'training.pt'
    torch.save(SimpleNamespace(model_config={'model_config': {'_target_': 'native'}, 'rank': 2, 'alpha': 1.0},
                               model_state_dict=model.state_dict(), ema_state_dict=ema.state_dict()), original)
    output, factors = tmp_path / 'uma.pt', tmp_path / 'uma_lora.pt'
    export_lora_checkpoint(original, output, factors)
    stock = torch.load(output, weights_only=False)
    assert stock.model_config == {'_target_': 'native'}
    ordinary = torch.optim.swa_utils.AveragedModel(radial_model())
    ordinary.load_state_dict(stock.ema_state_dict, strict=True)
    for name, module in targets.items():
        torch.testing.assert_close(stock.model_state_dict[f'{name}.weight'], module.weight)
        assert not torch.equal(stock.ema_state_dict[f'module.{name}.weight'], module.weight)
    restore_ema_lora(ordinary.module, factors)
    for name, value in ema.module.state_dict().items():
        torch.testing.assert_close(ordinary.module.state_dict()[name], value)


def test_uma_rejects_basis_for_different_model_split_or_conditioning(tmp_path):
    manifest = {'version': 1, 'task': 'omol', 'ntk': {}, 'charge': 0, 'spin': 1,
                'lora': True, 'train': {'frames': [{'id': 'frame-1'}]}}
    (tmp_path / 'training_split.json').write_text(json.dumps(manifest))
    (tmp_path / 'uma.pt').write_bytes(b'model')
    (tmp_path / 'uma_lora.pt').write_bytes(b'factors')
    _, _, provenance = _member_artifacts(tmp_path)
    np.savez(tmp_path / 'ntk_basis.npz', metadata=json.dumps({'fit_provenance': provenance,
                                                            'training_structure_count': 1}))
    with pytest.raises(ValueError, match='charge and spin'):
        load_uma_ntk_estimator(tmp_path, charge=1)
    (tmp_path / 'uma_lora.pt').write_bytes(b'other factors')
    with pytest.raises(ValueError, match='does not match'):
        load_uma_ntk_estimator(tmp_path)
