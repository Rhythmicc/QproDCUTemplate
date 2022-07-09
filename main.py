from dis import dis
from QuickProject.Commander import Commander
from QuickProject import QproDefaultConsole, QproInfoString, QproErrorString, _ask
from subprocess import Popen, PIPE
import pickle
import os

job_name = 'QproDCUTemplate'
default_sbatch = job_name
executable = f'dist/{job_name}'

with open('dist/last_id', 'r') as f:
    last_id = f.read()

if os.path.exists('dist/record'):
    with open('dist/record', 'rb') as f:
        record = pickle.load(f)
else:
    record = {}

roc_include = '/public/software/compiler/dtk-22.04/rocsparse/include'
roc_lib = '/public/software/compiler/dtk-22.04/rocsparse/lib'


def gflops_cal(ct: list):
    import re
    nnzCub = 0
    using_time = -1
    for line in ct:
        if line.startswith('marix A:'):
            nnzCub = int(line.strip().split()[-1])
            nnzCub *= int(nnzCub / 5.23)
        elif line.startswith('Device calculation finished!'):
            using_time = float(re.findall('\d+', line)[0])
    if nnzCub == 0 or using_time == -1:
        QproDefaultConsole.print(QproErrorString, '计算结果不完整')
        return -1
    return 2 * nnzCub / using_time / 1e3  # 依据A的行数计算的近似值


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


@app.command()
def compile(version: str = latest, gpu: bool = False):
    """
    编译项目
    :param version: 编译版本
    """
    with QproDefaultConsole.status("编译中"):
        code, content = external_EXEC(
            f"hipcc -Ofast -std=c++11 -I include -I kernel -I {roc_include} -L {roc_lib} {'-D gpu' if gpu else ''} -D VERSION='<{job_name}_v{version}.hpp>' -lomp -fopenmp -lrocsparse main.cpp -o {executable}", True)
    if code:
        QproDefaultConsole.print(QproErrorString, content.replace('errors', '[bold red]errors[/bold red]').replace(
            'warnings', '[bold yellow]warnings[/bold yellow]').replace('error', '[bold red]error[/bold red]').replace('warning', '[bold yellow]warning[/bold yellow]'))
    else:
        with open('dist/version', 'w') as f:
            f.write(version)
    return code, content


@app.command()
def run(batch: str = default_sbatch):
    """
    运行项目
    """
    with QproDefaultConsole.status(f'提交任务 "{batch}.sbatch" 中'):
        code, content = external_EXEC(f"sbatch < dist/{batch}.sbatch", True)
    if code:
        return
    job_id = content.split()[-1]
    QproDefaultConsole.print(QproInfoString, f"任务提交成功，任务ID：{job_id}")
    if _ask({
        'type': 'confirm',
        'name': 'confirm',
        'message': f'是否删除"log/{last_id}.loop"？',
        'default': True
    }):
        from QuickStart_Rhy import remove
        remove(f"log/{last_id}.loop")
    with open('dist/last_id', 'w') as f:
        f.write(job_id)


@app.command()
def compile_and_run(version: str = latest, gpu: bool = False, batch: str = default_sbatch):
    """
    编译并运行项目
    :param version: 编译版本
    """
    code, content = app.real_call('compile', version, gpu)
    if code:
        QproDefaultConsole.print(QproErrorString, '编译失败')
        return
    app.real_call('run', batch)


@app.command()
def status(job_id: str = last_id):
    """
    查看状态
    :param job_id: 任务ID
    """
    def show_log():
        if os.path.exists(f'log/{job_id}.loop'):
            from rich.panel import Panel
            with open(f'log/{job_id}.loop', 'r') as f:
                ct = f.read().strip()
            QproDefaultConsole.print(Panel(
                ct, title='[bold magenta]当前输出[/bold magenta]', width=QproDefaultConsole.width))

    with QproDefaultConsole.status("查询状态中"):
        code, content = external_EXEC(
            f"squeue --name={job_name}", without_output=True)
    if code:
        return
    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table(
        ['任务ID', '任务队列', '任务名称', '用户', '状态', '用时', '节点数目', '节点列表'], title='任务队列\n')
    items = [i.strip().split() for i in content.split('\n')[1:]]
    if not items:
        show_log()
        with open(f'log/{job_id}.loop', 'r') as f:
            ct = f.read().strip().split('\n')[1:]
            version = f'v{get_version()}'
            gflops = gflops_cal(ct)
            if ct[-2].startswith('Congratulation'):
                QproDefaultConsole.print(
                    QproInfoString, f'计算[bold green]通过[/bold green]，版本 "{version}" 性能：{gflops} GFlop/s')
                if version not in record:
                    record[version] = gflops
                    os.system(
                        f'cp kernel/{job_name}_{version}.hpp template/{job_name}_{version}.hpp')
                    QproDefaultConsole.print(
                        QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                    dump_record()
                elif gflops > record[version]:
                    record[version] = gflops
                    os.system(
                        f'cp kernel/{job_name}_{version}.hpp template/{job_name}_{version}.hpp')
                    QproDefaultConsole.print(
                        QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                    dump_record()
            else:
                QproDefaultConsole.print(
                    QproErrorString, f'计算[bold red]错误[/bold red]，版本 "{version}" 性能：{gflops} GFlop/s')
        return
    for item in items:
        table.add_row(*item)
    QproDefaultConsole.print(table, justify='center')
    QproDefaultConsole.print()
    show_log()


@app.command()
def cancel(job_id: str = last_id):
    """
    取消任务
    :param job_id: 任务ID
    """
    with QproDefaultConsole.status("取消任务中"):
        external_EXEC(f"scancel {job_id}")


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
    table = qs_default_table(['版本', '性能 (GFlops)', '最佳版本'], title=f'性能表\n')
    for item in sorted(list(record.keys()), key=lambda x: int(x[1:])):
        table.add_row(item, str(
            record[item]), '[bold green]√[/bold green]' if record[item] == max(record.values()) else '')
    QproDefaultConsole.print(table, justify='center')


if __name__ == "__main__":
    app()
