import pickle
import json
import os

job_name = 'QproDCUTemplate'
default_sbatch = job_name
executable = f'dist/{job_name}'

roc_include = '/public/software/compiler/dtk-22.04/rocsparse/include'
roc_lib = '/public/software/compiler/dtk-22.04/rocsparse/lib'

includePath = [
    roc_include,
    'include',
    'kernel',
]

libPath = [
    roc_lib
]

refresh_second = 5

performance_unit = 'GFlop/s'

def performance_cal(ct: list, last_batch: str = job_name):
    """
    在此自定义计算GFLOPS的方式
    
    :param ct: 日志文件的每行内容
    """
    return 0


def is_Success(ct: list, last_batch: str = job_name):
    """
    在此自定义判断是否成功

    :param ct: 日志文件的每行内容
    """
    return True


with open('dist/last_id', 'r') as f:
    _dt = json.load(f)
    last_id = _dt['last_id']
    last_batch = _dt['last_batch']

if os.path.exists('dist/record'):
    with open('dist/record', 'rb') as f:
        record = pickle.load(f)
else:
    record = {}


def get_version():
    with open('dist/version', 'r') as f:
        return f.read()


def latest_version():
    import os
    files = os.listdir('kernel')
    files = [int(f.split('.')[0].split('_v')[1])
             for f in files if f.endswith('.hpp')]
    return str(max(files))


def dump_record():
    with open('dist/record', 'wb') as f:
        pickle.dump(record, f)


latest = latest_version()