from src.operator_ui.startup_policy import legacy_startup_disabled


def test_generated_connected_r3_disables_legacy_startup_services():
    assert legacy_startup_disabled(
        {
            "OPERATOR_UI_CONNECTED_MODE": "1",
            "OPERATOR_UI_R3_PROFILE": "repository-v1",
            "OPERATOR_UI_DISABLE_LEGACY_STARTUP": "1",
        }
    )


def test_partial_or_default_off_configuration_keeps_legacy_startup_behavior():
    assert not legacy_startup_disabled({})
    assert not legacy_startup_disabled(
        {
            "OPERATOR_UI_CONNECTED_MODE": "0",
            "OPERATOR_UI_R3_PROFILE": "repository-v1",
            "OPERATOR_UI_DISABLE_LEGACY_STARTUP": "1",
        }
    )
    assert not legacy_startup_disabled(
        {
            "OPERATOR_UI_CONNECTED_MODE": "1",
            "OPERATOR_UI_R3_PROFILE": "repository-v1",
        }
    )
