def test_multichannel_module_import():
    from episurveil.inference.particle_filter import sir_filter_multichannel
    assert callable(sir_filter_multichannel)
