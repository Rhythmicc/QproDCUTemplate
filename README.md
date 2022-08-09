# QproDCUTemplate

曙光 DCU 集群的程序开发模板，提供基于 Qpro Commander 设计的命令工具集，通过`qrun --help`获取详细内容。_本模板需要 Qpro 0.9.14 或以上版本，请务必阅读完本文档后再开始使用。_

本文档图片中的所有`Qpro\DCUTemplate`字段在项目创建后都会被替换为`QproDCUTemplate`(相应项目名)。

## 工具集

![alt=''](https://cos.rhythmlian.cn/ImgBed/0d601753dd34a2a72fb21c0197343c3c.png)

### 注意

1. 必填参数需要按顺序填写。
2. 可选参数需要以`--<参数名> bala`方式修改默认值。
3. 上图中的`version`可选参数默认使用最新版本编译、`batch`选项默认为`QproDCUTemplate`、`job_id`选项默认为最新的运行 ID。
4. 上图中的`gpu`选项默认关闭，通过添加`--gpu`标志即可开启（无需赋值），开启`gpu`选项后，编译指令会增加`-D gpu`（表示在代码中添加`#define gpu`）。

### 样例

假设当前共有三个版本，我们希望**编译且运行**版本`2`，并且 batch 脚本指定为`dist/test.sbatch`，则命令为`qrun compile-and-run --version 2 --batch test`。

## 项目结构

![alt=''](https://cos.rhythmlian.cn/ImgBed/58c2cf88f36e1fdfdc7185c6aa5c8542.png)

| 文件/目录 | 含义                                                                                                     |
| :-------: | :------------------------------------------------------------------------------------------------------- |
|   dist    | 保存编译好的可执行文件 (QproDCUTemplate)、上次任务 ID (last_id)、当前可执行文件版本 (version)            |
|  include  | 必要的核函数编写需要的头文件，基础的工具集（可以在此实现无关紧要的 API）                                 |
|  kernel   | 保存实现的核函数不同版本，必须以`QproDCUTemplate_v<版本号>.hpp`命名                                      |
|    log    | 保存提交至任务队列的程序的输出                                                                           |
| template  | 保存性能最高的版本（通过解析输出的 log 文件），你可以修改`config/__init__.py`中的`gflops_cal`函数，自定义解析规则。 |
| config    | 配置表与辅助器                                                                                |

## 运行

1. **<font color='red'>第一次运行前</font>>: 修改`dist/QproDCUTemplate.sbatch`文件，将`你的任务队列名`替换为你的任务队列，填写`config/__init__.py`中的相关配置（包括性能计算函数、判定是否计算成功函数、includePath、libraryPath）。**
2. `qrun run`即可直接使用`dist/QproDCUTemplate.sbatch`文件作为任务提交，若你有其他`sbatch`文件，请将它拷贝至`dist`文件夹，通过`qrun run --batch <其他batch文件名>`方式调用。
3. 在`main.py`中，你可以修改`gflops_cal`和`is_Success`函数来自定义性能统计方式和计算结果是否成功的判定，传入的`ct: list`参数存储了当前任务日志文件的每行字符串。

## 高级

如果希望增加子命令，可以参照`config/main.py`文件中实现的以`@app.command()`装饰的函数自行设计。Qpro Commander 能够解析 int, float, str, list 四种类型的参数，函数体的注释用于描述此命令的含义和参数的含义。
