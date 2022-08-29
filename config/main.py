from QuickProject.Commander import Commander
from subprocess import Popen, PIPE
from . import *
from . import _status_show_log, _with_permission, _ask


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


def get_squeue(user: str = ''):
    name = f'{job_name}_{user}' if user else job_name
    code, content = external_EXEC(
        f"squeue --name={name}", without_output=True)
    if code:
        return -1
    return [i.strip().split() for i in content.split('\n')[1:]]


def realtime_output(file: str, user: str = ''):
    import time
    from QuickStart_Rhy.Wrapper import set_timeout

    @set_timeout(refresh_second)
    def get_content(p):
        return p.stdout.readline().strip()

    p = Popen(f'tail -F {file}', shell=True, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    while True:
        while line := get_content(p):
            QproDefaultConsole.print(line)
        time.sleep(refresh_second)
        q = get_squeue(user)
        if q == -1 or not q:
            break
    p.kill()


def lock(user: str = ''):
    import time
    with QproDefaultConsole.status('正在获取权限'):
        lock_path = f'dist/lock-{user}' if user else 'dist/lock'
        while os.path.exists(lock_path):
            time.sleep(2)
        with open(lock_path, 'w') as f:
            f.write('lock!')


def release(user: str = ''):
    lock_path = f'dist/lock-{user}' if user else 'dist/lock'
    external_EXEC(f'rm {lock_path}')


def show_log(job_id, user: str = ''):
    import time

    try:
        while not os.path.exists(f'log/{job_id}.loop'):
            time.sleep(refresh_second)
        realtime_output(f'log/{job_id}.loop', user)
    except KeyboardInterrupt:
        if _status_show_log and _ask({
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


def check_user(**kwargs):
    user = kwargs.get('user')
    if user and user not in users:
        QproDefaultConsole.print(QproErrorString, f'"{user}" 不是有效的用户')
        raise KeyboardInterrupt


@app.command()
def compile(version: str = latest, gpu: bool = False, user: str = users[0] if users else ''):
    """
    编译项目
    :param version: 编译版本
    """
    if not _with_permission:
        lock(user)
    job_path = f'dist/jobs_{user}.sh' if user else 'dist/jobs.sh'
    sbatch_path = f'dist/{user}.sbatch' if user else f'dist/{default_sbatch}.sbatch'
    version_path = f'dist/version-{user}' if user else 'dist/version'
    if not os.path.exists(job_path):
        with open(job_path, 'w') as f:
            f.write(f'#!/bin/bash\n\n')
        QproDefaultConsole.print(QproInfoString, f'用户 {user} 的任务文件已生成，请在 {job_path} 中添加任务后重新编译')
        exit(0)
    with QproDefaultConsole.status('生成任务文件中'):
        with open(job_path, 'r') as f:
            _ls = f.read().strip().split('\n')
            content = ''
            for line in _ls:
                if line.startswith('#'):
                    continue
                content += line + '\n'
        with open(sbatch_path, 'w') as f:
            from QuickProject import project_configure_path, dir_char
            project_path = dir_char.join(project_configure_path.split(dir_char)[:-1])

            print(f"""#!/bin/bash
#SBATCH -J {f'{job_name}_{user}' if user else job_name}
#SBATCH -p {queue_name}
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=90G
#SBATCH -o log/%j.loop
#SBATCH -e log/%j.loop

module unload compiler/rocm/3.3
module unload compiler/dtk/21.10.1
module load apps/anaconda3/5.2.0
module load compiler/dtk/22.04

cd {project_path}

echo [bold cyan]信息[/bold cyan] 编译中...

{hipcc} {hipcc_flags} -I {' -I '.join(includePath)} -L {' -L '.join(libPath)} {'-D gpu' if gpu else ''} -D VERSION='<{job_name}_v{version}.hpp>' -lomp -fopenmp -lrocsparse main.cpp -o {executable}

echo [bold cyan][信息][/bold cyan] 编译完成
echo [bold cyan][信息][/bold cyan] 开始运行

{content}""", file=f)
        with open(version_path, 'w') as f:
            f.write(version)
    if not _with_permission:
        release(user)


@app.command()
def run(user: str = users[0] if users else ''):
    """
    运行项目
    """
    global _status_show_log

    _status_show_log = True
    items = get_squeue(user)
    if items == -1:
        return
    if not _with_permission:
        lock(user)
    batch = user if user else default_sbatch
    with QproDefaultConsole.status(f'提交任务 "{batch}.sbatch" 中'):
        code, content = external_EXEC(f"sbatch < dist/{batch}.sbatch", True)
    if code:
        return
    job_id = content.split()[-1]
    QproDefaultConsole.print(QproInfoString, f"任务提交成功，任务ID：{job_id}")
    last_id, _ = get_last_id(user)
    if last_id == '-1' or _ask({
        'type': 'confirm',
        'name': 'confirm',
        'message': f'是否删除"log/{last_id}.loop"?',
        'default': True
    }):
        from QuickStart_Rhy import remove
        remove(f"log/{last_id}.loop")
    set_last_id(job_id, batch, user)
    if not _with_permission:
        try:
            app.real_call('status', job_id, user)
            release(user)
        except SystemExit:
            return
        except:
            release(user)
    else:
        app.real_call('status', job_id, user)


@app.command()
def compile_and_run(version: str = latest, gpu: bool = False, user: str = users[0] if users else ''):
    """
    编译并运行项目
    :param version: 编译版本
    :param gpu: 是否使用GPU
    :param batch: sbatch文件名
    """
    lock(user)
    try:
        global _with_permission
        _with_permission = True
        app.real_call('compile', version, gpu, user)
        app.real_call('run', user)
    except SystemExit:
        return
    except:
        QproDefaultConsole.print_exception()
        release(user)
    finally:
        release(user)


@app.command()
def status(job_id: str = '-1', user: str = users[0] if users else ''):
    """
    查看状态
    :param job_id: 任务ID，默认为-1，表示给定用户的最后一个任务
    :param user: 用户名
    """
    global cur_line_num
    cur_line_num = 0

    items = get_squeue(user)
    if items == -1:
        return
    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table(
        ['任务ID', '任务队列', '任务名称', '用户', '状态', '用时', '节点数目', '节点列表'], title='任务队列\n')
    for item in items:
        table.add_row(*item)
    QproDefaultConsole.print(table, justify='center')
    QproDefaultConsole.print()
    if job_id == '-1':
        job_id, _ = get_last_id(user)
    if _status_show_log:
        show_log(job_id, user)
    with open(f'log/{job_id}.loop', 'r') as f:
        ct = f.read().strip().split('\n')[1:]
        version = f'v{get_version()}'
        if is_Success(ct):
            performance = performance_cal(ct)
            QproDefaultConsole.print(
                QproInfoString, f'计算[bold green]通过[/bold green]，版本 "{version}" 指标：{performance} {performance_unit}')
            if user:
                return
            record = get_record()
            if version not in record:
                record[version] = performance
                import shutil
                shutil.copy(f'kernel/{job_name}_{version}.hpp',
                            f'template/{job_name}_{version}.hpp')
                QproDefaultConsole.print(
                    QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                dump_record(record)
            elif performance_cmp(performance, record[version]):
                record[version] = performance
                import shutil
                shutil.copy(f'kernel/{job_name}_{version}.hpp',
                            f'template/{job_name}_{version}.hpp')
                QproDefaultConsole.print(
                    QproInfoString, f'版本 "{version}" 的最佳实现已保存')
                dump_record(record)
        else:
            QproDefaultConsole.print(
                QproErrorString, f'计算[bold red]错误[/bold red]，版本 "{version}"')


@app.command()
def cancel(job_id: str = '-1', user: str = users[0] if users else ''):
    """
    取消任务
    :param job_id: 任务ID，默认为-1，表示给定用户的最后一个任务
    :param user: 用户名
    """
    if job_id == '-1':
        job_id, _ = get_last_id(user)
    with QproDefaultConsole.status("取消任务中"):
        external_EXEC(f"scancel {job_id}")
    release(user)


@app.command()
def reset(version: str):
    """
    重置版本记录
    :param version: 版本号
    """
    record = get_record()
    with QproDefaultConsole.status("重置版本中"):
        external_EXEC(f"rm -f template/{job_name}_v{version}.hpp")
        if f'v{version}' in record:
            record.pop(f'v{version}')
            dump_record(record)
        QproDefaultConsole.print(QproInfoString, f'版本 "v{version}" 已重置')


@app.command()
def show():
    """
    查看记录
    """
    from QuickStart_Rhy.TuiTools.Table import qs_default_table
    table = qs_default_table(
        ['版本', f'性能指标 ({performance_unit})', '最佳版本'], title=f'性能表\n')
    record = get_record()
    for item in sorted(list(record.keys()), key=lambda x: int(x[1:])):
        table.add_row(item, str(
            record[item]), '[bold green]√[/bold green]' if record[item] == performance_best(record.values()) else '')
    QproDefaultConsole.print(table, justify='center')


if __name__ == "__main__":
    app.bind_pre_call('compile', check_user)
    app.bind_pre_call('run', check_user)
    app.bind_pre_call('status', check_user)
    app.bind_pre_call('cancel', check_user)
    app()
