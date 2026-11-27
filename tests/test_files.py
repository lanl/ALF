"""ALF's file helpers (alframework.tools.files)."""
import pytest



def test_logged_output_copies_both_streams_and_an_uncaught_traceback(tmp_path, capsys):
    import sys
    from alframework.tools.files import logged_output
    path = tmp_path / 'training_log.txt'
    with pytest.raises(RuntimeError, match='boom'):
        with logged_output(path) as log:
            print('to stdout')
            print('to stderr', file=sys.stderr)
            log.write('only the log\n')
            raise RuntimeError('boom')
    text = path.read_text()
    assert 'to stdout' in text and 'to stderr' in text and 'only the log' in text
    assert 'Uncaught exception, logging interrupted.' in text and 'RuntimeError: boom' in text
    seen = capsys.readouterr()
    assert 'to stdout' in seen.out and 'to stderr' in seen.err and 'only the log' not in seen.out + seen.err


def test_working_directory_returns_and_creates_only_as_asked(tmp_path, monkeypatch):
    import os
    from alframework.tools.files import working_directory
    monkeypatch.chdir(tmp_path)
    with working_directory('new', create=True):
        assert os.getcwd() == str(tmp_path / 'new')
    assert os.getcwd() == str(tmp_path)
    with pytest.raises(FileExistsError):
        with working_directory('new', create=True):
            pass
    with pytest.raises(FileNotFoundError):
        with working_directory('missing', create=False):
            pass
    with working_directory('made'):
        pass
    assert (tmp_path / 'made').is_dir()
