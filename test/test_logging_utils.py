from loguru import logger

from fishtools.utils.logging import configure_cli_logging


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
