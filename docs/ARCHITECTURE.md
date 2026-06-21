# CRSEM 架构说明

## 1. 概述

CRSEM（寒区土壤侵蚀模型）是基于 RUSLE 方程框架的分布式土壤侵蚀模型，针对寒区特点增加了融雪侵蚀等过程。

**核心工作流程：**

```
数据准备 → 参数率定 → 模型运行 → 结果分析
```

**主要数据流：**

```
NetCDF 文件 → BasinDriver → RunContext → PreparedInputs → BaseModel.run_batch() → BatchRunResult
```

## 2. 核心原则

### 2.1 边界使用 xarray，内核使用 numpy

- I/O 与上下文层保留 `xarray`
- 模型内部统一使用 `numpy`
- 输出结果在边界重新包装为 `xarray`

### 2.2 单模型和集合模式统一

- 单模型运行：`ParameterBatch.n_members == 1`
- 集合运行：`ParameterBatch.n_members > 1`
- 系统中不再存在独立的 ensemble model

### 2.3 数据准备与模型运行解耦

- 数据准备模块独立于模型运行
- 通过标准 NetCDF 文件传递数据
- BasinDriver 只支持 NC 文件加载

## 3. 模块结构

```
CRSEM/
├── __init__.py
├── driver.py              # BasinDriver - 数据驱动
├── contracts.py           # RunContext, PreparedInputs, ParameterBatch
├── model.py               # ModelInputs, ModelOutputs, ModelRegistry
├── parameters.py          # CRSEMParameters, RUSLEParameters
├── _model_base.py         # BaseModel 抽象基类
├── _model_core.py         # 核心侵蚀计算函数
├── _model_crsem.py        # CRSEM 模型实现
├── _model_rusle.py        # RUSLE 模型实现
├── calibration_evaluation.py # 评价指标 (NSE, KGE, R² 等) 和罚函数
├── batch_runner.py        # run_parameter_batch()
├── preparation.py         # prepare_inputs()
├── result_aggregator.py   # ResultAggregator
├── calibrator.py          # Calibrator
├── calibration_api.py     # refine_parameters() API
├── calibration_optimizer.py
├── calibration_result.py
├── calibration_evaluation.py
├── calibration_reporting.py
├── data_preparation/      # 数据准备模块
│   ├── __init__.py
│   ├── spatial.py         # 空间处理工具
│   ├── builders.py        # NC 文件构建器
│   ├── quality.py         # 数据质量评估
│   └── obs_preprocessing.py # 观测数据预处理（日→月，缺测插补）
├── config.py              # 配置类
├── parameter_config.py    # 参数配置加载器
├── sensitivity.py         # 敏感性分析（OAT 参数敏感性、气候/NDVI 回归+SHAP）

scripts/
├── prepare_basin_drivers.py # 通用流域驱动数据准备
├── calibrate_parameters.py # 参数率定
├── run_model.py           # 模型运行
├── attribution_analysis.py # 驱动变量归因分析（real vs counterfactual，固定走 run_hillslope）
├── plot_ndvi_attribution_analysis.py # NDVI 归因结果三联图绘制
└── test_full_workflow.py  # 完整流程测试
```

## 4. 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    数据准备层 (Data Preparation)              │
│  data_preparation/spatial.py, builders.py, io_legacy.py    │
│  输出: static.nc, dynamic.nc, observations.nc              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    驱动层 (Driver Layer)                     │
│  BasinDriver.from_nc_files() → ModelInputs                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    上下文层 (Context Layer)                   │
│  RunContext, PreparedInputs                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    执行层 (Execution Layer)                  │
│  BaseModel.run_batch() → CRSEMModel / RUSLEModel           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    结果层 (Result Layer)                     │
│  BatchRunResult → ResultAggregator                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    率定层 (Calibration Layer)                │
│  Calibrator, Selector, ObjectiveEvaluator                  │
└─────────────────────────────────────────────────────────────┘
```

## 5. 数据流

### 5.1 数据准备

```
原始数据 (NetCDF/GeoTIFF/CSV)
        │
        ▼
data_preparation/spatial.py
  - align_to_basin()
  - apply_basin_mask()
        │
        ▼
data_preparation/builders.py
  - build_static_nc()    → static.nc (K, LS, IC, P_f, mask)
  - build_dynamic_nc()   → dynamic.nc (T, Pre, NDVI)
  - build_observations_nc() → observations.nc (Q, SSF)
```

### 5.2 模型运行

```
BasinDriver.from_nc_files(static_nc, dynamic_nc, observations_nc)
        │
        ▼
ModelInputs (xarray 容器)
        │
        ▼
to_run_context() → RunContext
        │
        ▼
prepare_inputs() → PreparedInputs (numpy 数组)
        │
        ▼
BaseModel.run_batch(source, ParameterBatch)
        │
        ▼
BatchRunResult
        │
        ▼
to_dataset() → xr.Dataset
```

### 5.3 校准运行流程

```
CalibrationModelRunner(source, output_mode)
        │
        ▼
prepare_inputs() → PreparedInputs (缓存，每批次只调用一次)
        │
        ▼
ModelFactory.create_model(model_type, params)
        │
        ▼
model._run_prepared_hillslope_river_numpy(prepared, context, output_mode)
        │
        ▼
CandidateEvaluation (损失函数计算)
```

**output_mode 参数说明：**

| 模式 | 输出内容 | 内存占用 | 适用场景 |
|------|----------|----------|----------|
| `full` | SSF_pred + A_channel + R_rain + R_melt + K/C/SDR_factor | 较高 | 详细诊断分析 |
| `compact` | SSF_pred + A_channel + R_rain + R_melt（K/C/SDR 设为 None） | 较低 | 加速校准 |

**Grid 模式下的空间平均：**

Grid 模式运行时，中间变量（R_rain, R_melt 等）在输出前进行空间平均（`np.nanmean`），用于诊断图绘制和罚函数计算。

## 6. 核心类说明

### 6.1 BasinDriver

数据驱动类，从预处理的 NetCDF 文件加载所有数据。

```python
driver = BasinDriver.from_nc_files(
    static_nc="example/tuotuohe_1990_2000/static.nc",
    dynamic_nc="example/tuotuohe_1990_2000/dynamic.nc",
    observations_nc="example/tuotuohe_1990_2000/observations.nc",
    station_name="沱沱河"
)
```

**职责：**
- 加载静态和动态空间数据
- 加载观测数据
- 提供流域元数据（面积、时间范围等）

### 6.2 ModelInputs

输入数据容器，存储 xarray DataArray。

```python
@dataclass
class ModelInputs:
    K: Optional[xr.DataArray]    # 土壤可蚀性因子
    LS: Optional[xr.DataArray]   # 地形因子
    IC: Optional[xr.DataArray]   # 连通性指数
    P_f: Optional[xr.DataArray]  # 水土保持措施因子
    T: Optional[xr.DataArray]    # 温度
    Pre: Optional[xr.DataArray]  # 降水量
    NDVI: Optional[xr.DataArray] # 植被指数
```

### 6.3 RunContext

运行上下文，包含模型运行所需的全部输入。

```python
@dataclass
class RunContext:
    inputs: ModelInputs
    q: pd.Series           # 流量观测
    ssf_obs: pd.Series     # 输沙量观测
    s_area: float          # 流域面积 (公顷)
    metadata: dict
```

### 6.4 ParameterBatch

参数批次容器，支持单成员和多成员。

```python
batch = ParameterBatch(
    values=np.array([[p1, p2, ...]]),  # shape: (n_members, n_params)
    param_names=("a_rain", "r_th", ...),
    weights=[0.5, 0.3, 0.2]  # 可选权重
)
```

### 6.5 BatchRunResult

模型运行结果，统一存储输出变量。

```python
result = run_parameter_batch("crsem", driver, params)
ssf_pred = result.variables["SSF_pred"]  # shape: (n_members, n_time)
ds = result.to_dataset()  # 转为 xr.Dataset
```

## 7. 模型参数

### 7.1 CRSEM 参数

| 参数 | 含义 | 默认值 | 范围 |
|------|------|--------|------|
| a_rain | 降雨侵蚀力系数 | 0.5 | [0.5, 1.0] |
| r_th | 降雨阈值 | 10.0 | [1, 20] |
| a_melt | 融雪侵蚀力系数 | 0.5 | [0.1, 1.0] |
| m_th | 融雪阈值 | 5.0 | [0, 10] |
| k_melt | 融雪系数 | 2.0 | [1, 5] |
| alpha_K | K因子调整系数 | 0.5 | [0.1, 0.8] |
| K_min_r | K因子最小比 | 0.5 | [0.4, 1.0] |
| K_max_r | K因子最大比 | 1.5 | [1.0, 2.0] |
| alpha_C | C因子系数 | 3.0 | [1, 5] |
| ic0 | SDR拐点IC值 | 0.5 | [0.1, 1.0] |
| k | SDR斜率参数 | 1.5 | [0.5, 4.0] |
| beta_sdr | SDR指数 | 0.5 | [0.3, 1.0] |
| c_base | 基础河道侵蚀 | 5.0 | [0.1, 20] |
| n_chan | 河道侵蚀指数 | 1.5 | [1.0, 2.0] |
| K_chan | 河道侵蚀系数 | 0.5 | [0.1, 1.0] |

## 8. 单位约定

| 变量 | 单位 |
|------|------|
| s_area | 公顷 (ha) |
| SSF | 吨/月 (t/month) |
| Q | 立方米/秒 (m³/s) |
| Pre | 毫米/月 (mm/month) |
| T | 摄氏度 (°C) |
| E_hillslope | 吨/公顷/月 (t/ha/month) |
| R_rain, R_melt | MJ·mm/(ha·h·month) |

## 9. 维护者指南

### 9.1 修改原则

- 涉及路径、读文件、坐标对齐：放 I/O 或 data_preparation 层
- 涉及公式、数组计算、侵蚀过程：放 Execution 层 (_model_*.py)
- 涉及 NSE、KGE、RMSE、罚项：放 Calibration 层
- 涉及 top-k、AIC、均值、分位数：放 Selector / Aggregator 层
- 涉及图表、日志、进度显示：放 Reporting 层

### 9.2 测试要求

- 所有新功能必须有单元测试
- 运行 `pytest tests/` 确保所有测试通过
- 测试覆盖率应保持在高水平

### 9.3 文档更新

- API 变更需更新 `docs/USER_GUIDE.md`
- 架构变更需更新本文档

## 10. 历史变更

| 版本 | 变更 |
|------|------|
| v1.0 | 初始版本，config-based 数据加载 |
| v2.0 | 添加 data_preparation 模块，支持 NC 文件模式 |
| v2.1 | BasinDriver 只支持 NC 文件模式，移除 config-based 加载 |
| v2.2 | 移除 dhesm 模块，移除 run/ 目录，脚本统一到 scripts/ |
| v2.3 | 添加 output_mode 参数控制校准输出详细程度；优化 batch runner 的 prepare_inputs 调用（每批次一次而非每成员一次）；CLI 新增 --run-mode 和 --plot-progress 参数，并按是否绘图自动选择 full/compact 输出 |
