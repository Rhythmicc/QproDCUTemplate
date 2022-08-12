from QuickProject.Commander import Commander
from QuickProject import QproDefaultConsole, QproInfoString, QproErrorString, _ask
from subprocess import Popen, PIPE
from . import *


app = Commander(True)


def external_EXEC(command: str, without_output: bool = False):
    res = Popen(command, shell=True, stdout=PIPE,
                stderr=PIPE, encoding='utf-8')
    ret_code = res.wait()
    content = res.communicate()[0].strip() + res.communicate()[1].strip()
    if ret_code and content and not without_output:
        QproDefaultConsole.print(QproErrorString, content)
    elif content and not without_output:
        QproDefaultConsole.print(QproInfoString, content)
    return ret_code, content


def get_squeue():
    code, content = external_EXEC(
        f"squeue --name={job_name}", without_output=True)
    if code:
        return -1
    return [i.strip().split() for i in content.split('\n')[1:]]


def lock():
    import time
    with QproDefaultConsole.status('正在获取权限'):
        while os.path.exists('dist/lock'):
            time.sleep(1)
        with open('dist/lock', 'w') as f:
            f.write('lock!')


def release():
    external_EXEC('rm dist/lock')


def show_log(job_id, kill_flag):
    def get_content(show_all: bool = True):
        global cur_line_num
        with open(f'log/{job_id}.loop', 'r') as f:
            if show_all:
                ct = f.read().strip()
            else:
                ct = f.readlines()
                _len = len(ct)
                ct = ''.join(ct[cur_line_num:])
                cur_line_num = _len
        return ct
    import time

    try:
        while not os.path.exists(f'log/{job_id}.loop'):
            time.sleep(refresh_second)

        while True:
            items = get_squeue()
            if items == -1:
                break
            ct = get_content(False).strip()
            if ct:
                QproDefaultConsole.print(ct)
            if not items:
                break
            time.sleep(refresh_second)
    except KeyboardInterrupt:
        if kill_flag and _ask({
            'type': 'confirm',
            'name': 'confirm',
            'message': '是否终止任务?',
            'default': False
        }):
            app.real_call('cancel', job_id)
            # exit(0)
            raise SystemExit(0)
    except:
        return


@app.command()
def compile(version: str = latest, gpu: bool = False, _with_permission: bool = False):
    """
    编译项目
    :param version: 编译版本
    """
    if not _with_permission:
        lock()
    with QproDefaultConsole.status('生成任务文件中'):
        with open('dist/jobs.sh', 'r') as f:
            content = f.read().strip()
        with open(f'dist/{default_sbatch}.sbatch', 'w') as f:
            print(f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=90G
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

module unload compiler/rocm/3.3
module unload compiler/dtk/21.10
module load apps/anaconda3/5.2.0
module load compiler/dtk/22.04

cd ${{HOME}}/{job_name}

{hipcc} -Ofast -std=c++11 -I {' -I '.join(includePath)} -L {' -L '.join(libPath)} {'-D gpu' if gpu else ''} -D VERSION='<{job_name}_v{version}.hpp>' -lomp -fopenmp -lrocsparse main.cpp -o {executable}

{content}""", file=f)
    if not _with_permission:
        release()


@app.command()
def run(_with_permission: bool = False):
    """
    运行项目
    """
    global last_id
    global last_batch
    items = get_squeue()
    if items == -1:
        return
    if not _with_permission:
        lock()
    batch = default_sbatch
    with QproDefaultConsole.status(f'提交任务 "{batch}.sbatch" 中'):
        code, content = external_EXEC(f"sbatch < dist/{batch}.sbatch", True)
    if code:
        return
    job_id = content.split()[-1]
    QproDefaultConsole.print(QproInfoString, f"任务提交成功，任务ID：{job_id}")
    if last_id == '-1' or _ask({
        'type': 'confirm',
        'name': 'confirm',
        'message': f'是否删除"log/{last_id}.loop"?',
        'default': True
    }):
        from QuickStart_Rhy import remove
        remove(f"log/{last_id}.loop")
    with open('dist/last_id', 'w') as f:
        json.dump({
            "last_id": job_id,
            "last_batch": batch
        }, f)
    if not _with_permission:
        try:
            app.real_call('status', job_id, True)
            release()
        except SystemExit:
            return
        except:
            release()
    else:
        app.real_call('status', job_id, True)


@app.command()
def compile_and_run(version: str = latest, gpu: bool = False):
    """
    编译并运行项目
    :param version: 编译版本
    :param gpu: 是否使用GPU
    :param batch: sbatch文件名
    """
    lock()
    try:
        app.real_call('compile', version, gpu, True)
        app.real_call('run', True)
    except SystemExit:
        return
    except:
        release()
    finally:
        release()


@app.command()
def status(job_id: str = last_id, _kill_flag: bool = False):
    """
    查看状态
    :param job_id: 任务ID
    :param real_time: 是否实时显示输出
    """
    global cur_line_num
    cur_line_num = 0

    items = get_squeue()
    if items == -1:
        return
    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table(
        ['任务ID', '任务队列', '任务名称', '用户', '状态', '用时', '节点数目', '节点列表'], title='任务队列\n')
    for item in items:
        table.add_row(*item)
    QproDefaultConsole.print(table, justify='center')
    QproDefaultConsole.print()
    show_log(job_id, _kill_flag)
    with open(f'log/{job_id}.loop', 'r') as f:
        ct = f.read().strip().split('\n')[1:]
        version = f'v{get_version()}'
        if is_Success(ct):
            performance = performance_cal(ct)
            QproDefaultConsole.print(
                QproInfoString, f'计算[bold green]通过[/bold green]，版本 "{version}" 指标：{performance} {performance_unit}')
            if version not in record:
                record[version] = performance
                import shutil
                shutil.copy(f'kernel/{job_name}_{version}.hpp',
                            f'template/{job_name}_{version}.hpp')
                QproDefaultConsole.print(
                    QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                dump_record()
            elif performance_cmp(performance, record[version]):
                record[version] = performance
                import shutil
                shutil.copy(f'kernel/{job_name}_{version}.hpp',
                            f'template/{job_name}_{version}.hpp')
                QproDefaultConsole.print(
                    QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                dump_record()
        else:
            QproDefaultConsole.print(
                QproErrorString, f'计算[bold red]错误[/bold red]，版本 "{version}"')


@app.command()
def cancel(job_id: str = last_id):
    """
    取消任务
    :param job_id: 任务ID
    """
    with QproDefaultConsole.status("取消任务中"):
        external_EXEC(f"scancel {job_id}")
    release()


@app.command()
def reset(version: str):
    """
    重置版本记录
    :param version: 版本号
    """
    with QproDefaultConsole.status("重置版本中"):
        external_EXEC(f"rm -f template/{job_name}_v{version}.hpp")
        if f'v{version}' in record:
            record.pop(f'v{version}')
            dump_record()
        QproDefaultConsole.print(QproInfoString, f'版本 "v{version}" 已重置')


@app.command()
def show():
    """
    查看记录
    """
    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table(
        ['版本', f'性能指标 ({performance_unit})', '最佳版本'], title=f'性能表\n')
    for item in sorted(list(record.keys()), key=lambda x: int(x[1:])):
        table.add_row(item, str(
            record[item]), '[bold green]√[/bold green]' if record[item] == performance_best(record.values()) else '')
    QproDefaultConsole.print(table, justify='center')


if __name__ == "__main__":
    app()
