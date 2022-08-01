import autoarray as aa


def test__settings_with_is_stochastic_true():

    settings = aa.SettingsPixelization(is_stochastic=False)
    settings = settings.settings_with_is_stochastic_true()
    assert settings.is_stochastic is True

    settings = aa.SettingsPixelization(is_stochastic=True)
    settings = settings.settings_with_is_stochastic_true()
    assert settings.is_stochastic is True
