# QproDCUTemplate

1. 基于 Qpro Commander 的编译、运行插件，通过`qrun --help`获取详细内容。
2. 项目结构:
   1. dist 目录: 保存编译好的可执行文件 (QproDCUTemplate)、上次任务 ID (last_id)、当前可执行文件版本 (version)
   2. include: 必要的核函数编写需要的头文件，基础的工具集（目前仅有前缀和）
   3. kernel : 保存实现的核函数不同版本，必须以`_v<版本号>.hpp`结尾
   4. log: 保存提交至任务队列后的程序输出
   5. template: 保存性能最高的版本（通过解析输出的 log 文件），你可以修改`main.py`中的`gflops_cal`函数，自定义解析规则。
