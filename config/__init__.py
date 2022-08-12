import pickle
import json
import os

job_name = 'QproDCUTemplate'
default_sbatch = job_name
executable = f'dist/{job_name}'
dtk_path = '/public/software/compiler/dtk-22.04'

hipcc = f'{dtk_path}/hip/bin/hipcc'
roc_include = f'{dtk_path}/rocsparse/include'
roc_lib = f'{dtk_path}/rocsparse/lib'

includePath = [
    roc_include,
    'include',
    'kernel',
]

libPath = [
    roc_lib
]

refresh_second = 5

performance_unit = 'GFlop/s' # 性能单位

def performance_cal(ct: list):
    """
    在此自定义计算性能的方式
    
    :param ct: 日志文件的每行内容
    """
    return 0


def performance_cmp(performance_now, performance_record) -> bool:
    """
    在此自定义比较两个性能的方式

    :param performance_now: 当前性能
    :param performance_record: 记录性能
    :return: 是否更好
    """
    return performance_now > performance_record


def performance_best(performance_list: list):
    """
    在此自定义计算最佳性能的方式

    :param performance_list: 所有性能列表
    """
    return max(performance_list)


def is_Success(ct: list):
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