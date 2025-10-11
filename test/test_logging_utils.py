from loguru import logger

from fishtools.utils.logging import configure_cli_logging, resolve_workspace_root


def test_configure_cli_logging_creates_workspace_log(tmp_path):
    component = "align"
    log_file = configure_cli_logging(tmp_path, component, console_level="INFO")

    try:
        logger.warning("hello world")
    finally:
        logger.remove()

    expected_root = tmp_path / "analysis" / "logs"
    assert log_file == expected_root / f"{component}.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "hello world" in content
    assert "WARNING" in content
    assert expected_root.is_dir()


def test_resolve_workspace_root_uses_workspace_detection(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ROOT.DONE").write_text("ok")
    target = workspace / "analysis" / "deconv" / "registered--roi+cb"
    target.mkdir(parents=True)

    root, has_analysis = resolve_workspace_root(target)

    assert root == workspace
    assert has_analysis


def test_resolve_workspace_root_falls_back_without_done(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "analysis" / "deconv").mkdir(parents=True)

    root, has_analysis = resolve_workspace_root(workspace / "analysis" / "deconv")

    assert root == workspace
    assert has_analysis
