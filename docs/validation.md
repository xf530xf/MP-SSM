# 实测验证记录

验证日期：2026-09-16。所有下列已运行项目均在当前 Mac 本地完成。

## 环境

| 项目 | 实测值 |
| --- | --- |
| 系统 | macOS 15.3.1 / ARM64 |
| Python | 3.14.5 |
| PyTorch | 2.10.0 |
| NumPy | 2.4.1 |
| OpenCV | 5.0.0（安装包 5.0.0.93） |
| pytest | 9.1.1 |
| CUDA | 不可用，无 NVIDIA GPU |
| 实际验证设备 | CPU，reference 选择性扫描 |

虚拟环境位于项目 `.venv`；`pip check` 通过。`pip install -e . --no-deps` 已成功，`mp-ssm --help` 和 `python -m mp_ssm --help` 均可用。本机环境清单见 `runs/verification/environment.lock.txt`；它不是服务器 CUDA 环境的安装锁文件。

## 自动化验收

执行：

```bash
.venv/bin/python -m pytest -q --junitxml=runs/verification/pytest.xml
```

最终结果：**32 passed，1 skipped，0 failed**。

已验证：

- 六条扫描路线在正方形、长方形、单行、单列上的完整覆盖与逆变换。
- 可手算标量 SSM 递推输出及解析输入梯度。
- 输入相关投影、A、D、Δ 参数的有限、非零梯度。
- 奇数及非方形输入恢复原尺寸，完整默认网络的小尺寸前向与反向。
- 多尺度对齐以及三个分支的梯度流。
- 同一进程先推理、再训练时，跨模块和设备索引缓存可安全参与反向传播。
- 0/1、0/255 mask，错误灰度标签、缺失配对与尺寸不匹配。
- 同步翻转和 MixUp 软标签；按完整视频划分及跨集合来源冲突检查。
- 手算混淆矩阵、空目标/预测、双类别 mIoU、流式 ODS/OIS 与逐阈值直接计算相符。
- 连续两轮训练与第一轮后断点恢复，最终全部模型状态逐位相同；恢复至原目录或新目录均通过。
- 训练、保存、加载、评估、原尺寸图片预测、连通区域坐标、视频编码及重新解码。
- benchmark 延迟/吞吐输出、部分 FLOPs 标记及大尺寸参考后端保护。

跳过项：`test_cuda_reference_forward_and_backward`，原因是本机没有 NVIDIA CUDA。该测试已提供，在服务器安装 CUDA 扩展后可执行。

## 保留的可检查产物

| 产物 | 位置 | 含义 |
| --- | --- | --- |
| JUnit 测试报告 | `runs/verification/pytest.xml` | 自动化测试通过/跳过明细 |
| 汇总 JSON | `runs/verification/summary.json` | 环境、状态、默认模型参数统计 |
| 默认模型统计 | `runs/verification/paper_profile.json` | 正式结构的实际参数和存储字节 |
| 合成输入 | `data/synthetic` | 4 张训练、2 张验证、2 张测试；带合成标记 |
| 合成训练 | `runs/smoke/history.jsonl` | 第 1 轮后恢复至第 2 轮，共 8 步 |
| 合成检查点 | `runs/smoke/latest.pt`、`best.pt` | 仅为流程演示，不能用于实际微塑料检测 |
| 合成评估 | `runs/smoke/test_metrics.json` | 2 张合成测试图，非论文精度 |
| 图片输出 | `runs/smoke/predictions` | 原尺寸概率数组、mask、叠加图和区域 JSON |
| 视频输出 | `runs/smoke/video_predictions` | 2 帧叠加视频，已重新解码验证 |

合成训练仅两轮；默认阈值下测试预测仍为空前景。其前景 F1=0，不具有实际检测效果。损失下降和各入口跑通只能证明工程流程可执行，不能证明真实数据精度。

## 默认结构参数实测

使用 `configs/paper.yaml` 架构、CPU reference 后端执行 `--profile-only`：

| 指标 | 实测值 |
| --- | ---: |
| 可训练参数 | 416,585 |
| FP32 参数张量字节 | 1,666,340 |
| Buffer 张量字节 | 5,592 |
| PyTorch 序列化 state_dict 字节 | 1,772,453 |

序列化体积含格式元数据；完整训练检查点还包含优化器和随机状态，因此明显更大。该配置并不等于文稿的 950,000 参数。

另运行了 64×64、24,507 参数的 **smoke 小模型** CPU benchmark，原始结果保存在 `runs/verification/smoke_benchmark.json`。这不是正式 MP-SSM 的 512/2048 性能测试，不用于支持文稿中的实时声明。

## 尚未验证

- NVIDIA 环境安装、CUDA/参考算子的实际数值对照和 CUDA 混合精度。
- 真实 MP-NTNU 数据、视频来源元数据及完整 200 轮训练。
- 正式网络 512/2048 的 CUDA 延迟、峰值显存、推理吞吐与端到端视频速度。
- MPS 性能、边缘设备部署、相机采集和跨帧唯一粒子计数。
- 文稿所报精度、0.95M 参数、0.96G 总 FLOPs、65 FPS、内存与权重体积。

服务器验证入口和数据协议见项目 README；所有未验证项目均没有被填为通过。
