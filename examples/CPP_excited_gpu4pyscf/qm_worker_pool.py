"""Start ordinary Parsl QM workers bound to allocated GPU UUIDs.

GPU4PySCF's single-GPU kernels require local CUDA device zero. Resolve the
scheduler-visible devices in a short-lived CUDA process, then let Parsl bind
each fresh worker to one UUID before importing any calculation code. The
manager's inherited mask stays unchanged, including its order/subset.
"""
import json
import runpy
import subprocess
import sys


CUDA_UUID_QUERY = '''import json, uuid, cupy
runtime = cupy.cuda.runtime
print(json.dumps(["GPU-" + str(uuid.UUID(bytes=runtime.getDeviceProperties(i)["uuid"]))
                  for i in range(runtime.getDeviceCount())]))
'''


def visible_accelerators(query=None):
    if query is None:
        query = lambda: subprocess.check_output(
            [sys.executable, '-c', CUDA_UUID_QUERY], text=True, timeout=60)
    values = json.loads(query())
    if (not isinstance(values, list) or not values or len(values) != len(set(values))
            or any(not isinstance(v, str) or not v.startswith('GPU-') for v in values)):
        raise RuntimeError('Cannot bind QM workers: expected distinct allocated GPU UUIDs')
    return values


def worker_arguments(arguments, accelerators):
    # This wrapper owns binding; reject explicit names that could bypass the
    # scheduler-visible set. The executor template supplies an empty flag.
    flag = '--available-accelerators'
    arguments = list(arguments)
    if flag in arguments:
        index = arguments.index(flag)
        if any(not a.startswith('-') for a in arguments[index+1:]):
            raise ValueError('QM accelerator binding must come from inherited CUDA visibility')
        arguments.pop(index)
    return arguments + [flag, *accelerators]


def main():
    sys.argv = ['process_worker_pool.py', *worker_arguments(sys.argv[1:], visible_accelerators())]
    runpy.run_module('parsl.executors.high_throughput.process_worker_pool', run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
