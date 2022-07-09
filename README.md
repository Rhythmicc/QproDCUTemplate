# QproDCUTemplate

曙光DCU集群的程序开发模板，提供基于 Qpro Commander 设计的命令工具集，通过`qrun --help`获取详细内容。*本模板需要Qpro 0.9.14或以上版本，请务必阅读完本文档后再开始使用。*

## 工具集

![](https://cos.rhythmlian.cn/ImgBed/6554a3b65224a4a6187543b33e7fe16f.png)

### 注意

1. 必填参数需要按顺序填写。
2. 可选参数需要以`--<参数名> bala`方式修改默认值。
3. 上图中的`version`可选参数默认使用最新版本编译、`batch`选项默认为`QproDCUTemplate`、`job_id`选项默认为最新的运行ID。
4. 上图中的`gpu`选项默认关闭，通过添加`--gpu`标志即可开启（无需赋值），开启`gpu`选项后，编译指令会增加`-D gpu`（表示在代码中添加`#define gpu`）。

### 样例

假设当前共有三个版本，我们希望**编译且运行**版本`2`，并且batch脚本指定为`dist/test.sbatch`，则命令为`qrun compile-and-run --version 2 --batch test`。

## 项目结构

![](https://cos.rhythmlian.cn/ImgBed/58c2cf88f36e1fdfdc7185c6aa5c8542.png)

注意：上图中以`Qpro DCUTemplate`开头的文件名，将在项目创建时自动被修改为`项目名`。

| 文件/目录 | 含义                                                         |
| :-------: | :----------------------------------------------------------- |
|   dist    | 保存编译好的可执行文件 (QproDCUTemplate)、上次任务 ID (last_id)、当前可执行文件版本 (version) |
|  include  | 必要的核函数编写需要的头文件，基础的工具集（可以在此实现无关紧要的API） |
|  kernel   | 保存实现的核函数不同版本，必须以`QproDCUTemplate_v<版本号>.hpp`命名 |
|    log    | 保存提交至任务队列的程序的输出                               |
| template  | 保存性能最高的版本（通过解析输出的 log 文件），你可以修改`main.py`中的`gflops_cal`函数，自定义解析规则。 |

## 运行

1. 第一次运行前: 修改`dist/QproDCUTemplate.sbatch`文件，将`你的任务队列名`替换为你的任务队列。
2. `qrun run`即可直接使用`dist/QproDCUTemplate.sbatch`文件作为任务提交，若你有其他`sbatch`文件，请将它拷贝至`dist`文件夹，通过`qrun run --batch <其他batch文件名>`方式调用。