import numpy as np
import pytest
from alframework.tools.force_screen import physical_force_diagnostics, physical_force_exceeds, validated_force_cutoff


def test_cutoff_uses_atom_norm_of_member_mean_and_includes_other_states():
    forces=np.zeros((2,2,1,3));forces[:,0,0]=[3.,4.,0.];forces[:,1,0]=[17.,0.,0.]
    pred=dict(state_energies=np.zeros((2,2,1)),state_forces=[forces],
              forces=[forces[:,0]],propagation_forces=np.zeros((1,3)))
    metrics=physical_force_diagnostics(pred)
    assert metrics['physical_F0_max']==5. and metrics['Fmeanmax']==17.
    assert physical_force_exceeds(metrics,16.)
    # Opposing member errors are uncertainty, not a high ensemble-mean force.
    pred=dict(forces=[np.array([[[100.,0,0]],[[-100.,0,0]]])],propagation_forces=np.full((1,3),1e9))
    assert physical_force_diagnostics(pred)['Fmeanmax']==0.
    assert not physical_force_exceeds(physical_force_diagnostics(pred),16.)


def test_ground_state_boundary_and_nonfinite_force():
    pred=dict(forces=[np.array([[[3.,4.,0.]]])])
    assert not physical_force_exceeds(physical_force_diagnostics(pred),5.)
    assert physical_force_exceeds(physical_force_diagnostics(pred),4.999)
    pred['forces'][0][0,0,0]=np.nan
    assert physical_force_exceeds(physical_force_diagnostics(pred),16.)
    with pytest.raises(ValueError,match='schema/source'):
        physical_force_exceeds({'Fmeanmax':1.},16.)


@pytest.mark.parametrize('value',[0,-1,True,np.inf,np.nan])
def test_force_cutoff_configuration(value):
    with pytest.raises(ValueError,match='finite and positive'):
        validated_force_cutoff(value)
