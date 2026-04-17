from main import missing_packages, run_demo_mode


def test_missing_packages_returns_list():
    out = missing_packages()
    assert isinstance(out, list)


def test_run_demo_mode_writes_outputs():
    result = run_demo_mode()
    assert result["mode"] == "demo"
    assert len(result["outputs"]) == 3
