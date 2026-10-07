from adapter_scale import apply_adapter_scale, reset_adapter_scale, trained_scales


class Mod:
    def __init__(self, scaling):
        self.scaling = scaling


class Model:
    def __init__(self, mods):
        self._m = mods

    def modules(self):
        return iter(self._m)


def test_scale_and_reset_idempotent():
    m = Model([Mod({"default": 4.0}), Mod({})])
    assert apply_adapter_scale(m, 0.5) == 1
    assert m._m[0].scaling["default"] == 2.0
    apply_adapter_scale(m, 0.5)                       # nao acumula
    assert m._m[0].scaling["default"] == 2.0
    reset_adapter_scale(m)
    assert m._m[0].scaling["default"] == 4.0


def test_adapter_added_after_first_call_is_scaled_from_its_own_base():
    mod = Mod({"a": 4.0})
    m = Model([mod])
    apply_adapter_scale(m, 0.5)
    mod.scaling["b"] = 8.0                            # 2o adapter carregado depois
    apply_adapter_scale(m, 0.5)
    assert mod.scaling == {"a": 2.0, "b": 4.0}
    assert trained_scales(m) == {"a": 4.0, "b": 8.0}


def test_only_named_adapter():
    mod = Mod({"a": 4.0, "b": 8.0})
    m = Model([mod])
    apply_adapter_scale(m, 0.5, adapter="b")
    assert mod.scaling == {"a": 4.0, "b": 4.0}
