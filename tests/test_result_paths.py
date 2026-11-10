"""Result paths must remain usable when a Forge caller has a different cwd."""
from pathlib import Path

from config import load_config


def test_result_and_record_paths_survive_a_different_client_cwd(tmp_path, monkeypatch):
    node_dir = tmp_path / 'node'
    node_dir.mkdir()
    config = node_dir / 'config.yaml'
    config.write_text('dataset_path: /dataset\nsuite: lh_mtlc\n'
                      'action_mode: abs_euler\npolicy_id: calvin_policy\n'
                      'result_dir: results\nrecord_dir: recordings\n')
    monkeypatch.chdir(node_dir)
    cfg = load_config(config)
    result = Path(cfg.result_dir) / 'batch.json'
    result.parent.mkdir()
    result.write_text('{}')
    monkeypatch.chdir(tmp_path)
    assert Path(cfg.result_dir).is_absolute()
    assert (Path(cfg.result_dir) / 'batch.json').read_text() == '{}'
    assert Path(cfg.record_dir) == node_dir / 'recordings'
