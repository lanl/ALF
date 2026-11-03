"""ALCHEMI subprocess entrypoint: physics stays in its qualified Python 3.11 env."""
from pathlib import Path
import pickle
import sys
import traceback

from campaign_tasks import unwrap
import campaign_tasks
from campaign_support import atomic_json


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    try:
        with (directory / 'input.pkl').open('rb') as handle:
            request = pickle.load(handle)
        if request['function'] not in ('sample_many', 'prediction_check'):
            raise ValueError('Unsupported sampler backend operation')
        result = unwrap(getattr(campaign_tasks, request['function']).func(*request['args']))
        with (directory / 'output.pkl').open('wb') as handle:
            pickle.dump(result, handle, protocol=4)
    except BaseException as exc:
        atomic_json(directory / 'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
