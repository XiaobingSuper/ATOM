import pytest

from atom.models.kimi_k3.mono.kernels.stages import energy_gfx1250 as energy
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tk


def test_the_default_waits_carry_no_suffix(monkeypatch):
    monkeypatch.delenv("ATOM_KIMI_MONO_ENERGY", raising=False)
    assert energy.suffix() == "" and not energy.lead()


def test_every_switch_lands_in_the_suffix(monkeypatch):
    monkeypatch.setenv("ATOM_KIMI_MONO_ENERGY", "wait8.ar32.lead")
    assert energy.suffix() == "_ena32lw8" and energy.lead()


@pytest.mark.parametrize("value", ("wait0", "ar128", "sleep4", "lead2"))
def test_a_bad_switch_is_refused(monkeypatch, value):
    monkeypatch.setenv("ATOM_KIMI_MONO_ENERGY", value)
    with pytest.raises(ValueError):
        energy.suffix()


@pytest.mark.parametrize("value", ("wait8", "ar32"))
def test_a_switched_all_reduce_is_a_build_of_its_own(monkeypatch, value):
    monkeypatch.delenv("ATOM_KIMI_MONO_ENERGY", raising=False)
    base = tk.build_k2_tail(8, 40, 1e-5, identity_routes=True).name
    gather = ar.build_gather_allreduce(8, 40).name
    monkeypatch.setenv("ATOM_KIMI_MONO_ENERGY", value)
    assert tk.build_k2_tail(8, 40, 1e-5, identity_routes=True).name == base + energy.suffix()
    assert ar.build_gather_allreduce(8, 40).name.endswith(energy.suffix())
    assert ar.build_gather_allreduce(8, 40).name != gather
