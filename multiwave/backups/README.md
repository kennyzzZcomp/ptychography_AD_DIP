# 共同 OPD／独立吸收方向备份

opd_spectral_20260929_150116.zip 是切换到零相位共享 USAF 之前的完整研究快照。

- 包含 82 个文件：multiwave 代码、文档、Notebook、测试、3 组本地实验结果，以及导入的 U-Net 依赖源文件。
- 不含可重新生成的 __pycache__。
- ZIP 完整性检查通过，并对所有源文件的 SHA-256 逐一校验。
- BACKUP_MANIFEST.json 记录文件校验和，RESTORE.txt 记录恢复方式。

恢复时解压到独立空目录，从解压根目录运行 python -m multiwave.run_simulation。不要直接覆盖当前项目。该备份是旧 idea 的保留版本，当前工作继续在 multiwave 主目录推进共享纯振幅 USAF。

