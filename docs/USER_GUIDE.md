# CRSEM 模型使用说明

## 1. 概述

CRSEM 是一个半分布式的月时间尺度，1km 空间尺度的寒区土壤侵蚀模型（Cold Region Soil Erosion Model）。
模型的产沙部分基于 RUSLE 方程框架，针对寒区特点增加了融雪侵蚀、冻融过程。同时接入基于供应能力的河道输沙模型，使得坡面
的产沙汇入流域出口，从而使用水文站观测的径流 Q 和悬移质泥沙通量 Q 进行率定。

### 1.1 模型物理机制

#### 1.1.1 坡面产沙计算

模型采用 RUSLE 方程计算坡面土壤侵蚀量：

```
A = R × K × LS × C × P_f
```

其中：
- **A**: 坡面侵蚀量 (t/ha/month)
- **R**: 降雨/融雪侵蚀力因子 (MJ·mm/(ha·h·month))
- **K**: 土壤可蚀性因子 (t·ha·h/(ha·MJ·mm))
- **LS**: 地形因子 (无量纲)
- **C**: 植被覆盖因子 (无量纲)
- **P_f**: 水土保持措施因子 (无量纲，默认为 1)

#### 1.1.2 降雨和融雪侵蚀力

降水根据月平均气温自动 partitioning，然后分别计算侵蚀力：

```
降雨/降雪分离：
  P_rain = P    (当 T > T_threshold)
  P_snow = P    (当 T ≤ T_threshold)

降雨侵蚀力：R_rain = max(0, a_rain × (P_rain - r_th))

融雪侵蚀力：R_melt = max(0, a_melt × (M_actual - m_th))
```

其中：
- `T_threshold`: 降雨/降雪温度阈值 (默认 0°C)
- `a_rain, a_melt`: 侵蚀力系数
- `r_th, m_th`: 产流阈值 (mm/month)

#### 1.1.3 融雪计算（度日法）

积雪累积和消融采用度日因子法：

```
积雪累积：S_pack = S_pack + P_snow

潜在融雪量：M_potential = k_melt × max(0, T - T_melt) × N_days

实际融雪量：M_actual = min(S_pack, M_potential)

积雪更新：S_pack = S_pack - M_actual
```

其中：
- `k_melt`: 度日融雪因子 (mm/°C/day)
- `T_melt`: 融雪温度阈值 (默认 2°C)
- `N_days`: 当月天数

#### 1.1.4 冻融侵蚀效应

冻融作用通过增强土壤可蚀性来体现，采用高斯函数描述温度对冻融效应的影响：

```
K = K_base × (1 + alpha_K × F_i)

F_i = exp(-(T - T_0)² / (2 × sigma_K²))
```

其中：
- `K_base`: 基础土壤可蚀性
- `alpha_K`: 冻融增强系数
- `T_0`: 冻融效应峰值温度 (默认 0°C)
- `sigma_K`: 冻融温度窗口宽度 (默认 2.5°C)

冻融效应在 0°C 附近最强，当温度远离冰点时迅速减弱。K 值被限制在 `[K_min, K_max]` 范围内。


#### 1.1.5 植被覆盖因子 (C 因子)

采用 Van der Knijff (2000) 公式从 NDVI 计算 C 因子：

```
C = exp(-alpha_C × NDVI / (1 - NDVI))
```

NDVI 被限制在 [0.05, 0.95] 范围内以避免数值问题。

#### 1.1.6 泥沙输移比 (SDR)

采用动态 SDR 模型，考虑连通性指数和降水影响：

```
基础 SDR: SDR_base = 0.8 / (1 + exp((ic0 - IC) / k))

动态调整因子：f_dyn = clip(1 + beta_sdr × (P_total / P_mean), 1, 3)

最终 SDR: SDR = min(1, SDR_base × f_dyn)
```

其中：
- `IC`: 连通性指数
- `ic0`: SDR 拐点 (此时 SDR=0.4)
- `k`: SDR 曲线斜率参数
- `beta_sdr`: 降水动态调整系数
- `P_total = P_rain + M_actual`
- `P_mean`: 参考降水量 (默认 50mm)

#### 1.1.7 河道输沙与侵蚀/沉积

河道部分采用输运能力模型：

```
河道输运能力：T_cap = c_base × Q^n_chan

侵蚀势能：E_potential = T_cap - S_in

河道侵蚀/沉积：
  - 当 E_potential > 0: A_channel = E_potential × K_chan  (河道侵蚀)
  - 当 E_potential ≤ 0: A_channel = E_potential          (河道沉积)
```

其中：
- `Q`: 河道流量 (m³/s)
- `S_in = E_hillslope × SDR × S_area`: 坡面来沙量 (t)
- `c_base`: 基础输运系数
- `n_chan`: 流量指数 (通常 1-2)
- `K_chan`: 河道可蚀性系数

**物理机制解释：**
- **当 T_cap > S_in 时**：河道有多余的输运能力，可以进一步侵蚀河床/河岸，发生**河道侵蚀**
- **当 T_cap < S_in 时**：来沙量超过河道输运能力，多余泥沙沉积，发生**河道沉积**

#### 1.1.8 流域出口总输沙量

```
SSF_pred = (E_hillslope × SDR × S_area) + A_channel
```

其中 `S_area` 为流域面积 (ha)。

#### 1.1.9 模型参数汇总

| 参数 | 默认范围 | 物理意义 |
|------|---------|---------|
| a_rain | [0.5, 1.0] | 降雨侵蚀力系数 |
| r_th | [1, 20] | 降雨侵蚀阈值 (mm) |
| a_melt | [0.1, 1.0] | 融雪侵蚀力系数 |
| m_th | [0, 10] | 融雪侵蚀阈值 (mm) |
| k_melt | [1, 5] | 度日融雪因子 (mm/°C/day) |
| alpha_K | [0.1, 0.8] | 冻融增强系数 |
| K_min_r | [0.4, 1.0] | K 因子最小比例 |
| K_max_r | [1.0, 2.0] | K 因子最大比例 |
| alpha_C | [1, 5] | NDVI-C 因子转换系数 |
| ic0 | [0.1, 1.0] | SDR 拐点 IC 值 |
| k | [0.5, 4.0] | SDR 曲线斜率 |
| beta_sdr | [0.3, 1.0] | SDR 降水动态系数 |
| c_base | [0.1, 20] | 河道输运基础系数 |
| n_chan | [1.0, 2.0] | 河道流量指数 |
| K_chan | [0.1, 1.0] | 河道可蚀性系数 |

**注意：** 参数的默认值、边界和惩罚设置可以通过配置文件进行自定义。详见第 3.2 节 "使用配置文件管理参数"。

---

**核心工作流程：**

```
数据准备 → 参数率定 → 模型运行 → 结果分析
```

**主要脚本：**

| 脚本 | 位置 | 功能 |
|------|------|------|
| 数据准备 | `scripts/prepare_basin_drivers.py` | 通用流域驱动数据制备 |
| 参数率定 | `scripts/calibrate_parameters.py` | 集合参数优化 |
| 模型运行 | `scripts/run_model.py` | 集合模拟输出 |
| 结果绘图 | `scripts/plot_ssf_comparison.py` | 模拟观测对比图 |

## 2. 数据准备

### 2.1 输入数据要求

模型需要三类预处理的 NetCDF 文件：

| 文件 | 变量 | 说明 |
|------|------|------|
| `static.nc` | K, LS, IC, P_f, mask | 静态空间数据 |
| `dynamic.nc` | T, Pre, NDVI | 时变驱动数据 |
| `observations.nc` | Q, SSF | 观测数据（率定时必需） |

**变量说明：**

static.nc
- **K**: 土壤可蚀性因子 (t·ha·h/(ha·MJ·mm))
- **LS**: 地形因子 (无量纲)
- **IC**: 连通性指数 (无量纲)
- **P_f**: 水土保持措施因子 (无量纲，默认为 1)

dynamic.nc
- **T**: 温度 (°C)
- **Pre**: 降水量 (mm/月)
- **NDVI**: 归一化植被指数 [0, 1]

observation.nc
- **Q**: 流量 (m³/s)
- **SSF**: 悬移质输沙量 (吨/月)

### 2.2 通用流域数据准备脚本

项目提供通用的流域数据准备脚本 `scripts/prepare_basin_drivers.py`，支持通过配置文件管理多个流域。

**配置文件：**

- `config/basin_data_sources.tuotuohe.yml`: 定义流域原始数据源路径（Windows 路径自动转换为 WSL 路径）
- `config/prepared_datasets.example.yml`: 展示已制备数据路径的配置格式示例

**基本用法：**

```bash
python scripts/prepare_basin_drivers.py \
  --config config/basin_data_sources.tuotuohe.yml \
  --basin tuotuohe \
  --years 1990 2000 \
  --output example/tuotuohe_1990_2000
```

**带质量报告：**

```bash
python scripts/prepare_basin_drivers.py \
  --config config/basin_data_sources.tuotuohe.yml \
  --basin tuotuohe \
  --years 1990 2000 \
  --output example/tuotuohe_1990_2000 \
  --quality-report
```

**参数说明：**

| 参数 | 含义 | 默认值 | 必需 | 备注 |
|------|------|--------|------|------|
| `--config` | 数据源配置 YAML 路径 | 无 | 是 | 定义流域输入数据位置 |
| `--basin` | 流域名称 | 无 | 是 | 必须与配置文件中的流域键一致 |
| `--years START END` | 起止年份 | `1990 2000` | 否 | 两个整数，起止年份均包含在处理范围内 |
| `--output` | 输出目录 | 无 | 是 | 生成 `drivers/`、元数据和可选质量报告 |
| `--quality-report` | 生成并打印质量报告 | `False` | 否 | 额外写出 `quality_report.json` |
| `--verbose`, `-v` | 打印详细过程信息 | `False` | 否 | 便于排查数据准备问题 |

**配置文件格式 (`config/basin_data_sources.tuotuohe.yml`)：**

```yaml
basins:
  tuotuohe:
    name: "沱沱河"
    basin_template: "/mnt/d/code/DH_CRSEM/prepared_inputs/tuotuohe_1990_2000/basin_static.nc"
    observation_csv: "/mnt/h/datasets/观测数据/水沙/TTH_imputed_final.csv"

    static:
      k_file: "/mnt/g/RTS_route/file/k_factor.nc"
      ls_file: "/mnt/g/RTS_route/file/ls_factor.nc"
      ic_file: "/mnt/g/RTS_route/file/ic_factor.tif"

    dynamic:
      meteo_file: "/mnt/g/RTS_route/driver/Meteo_merged_SRYaR_monthly_adj.nc"
      ndvi_file: "/mnt/g/RTS_route/driver/NDVI_merged_SRYaR_monthly_adj1.nc"

    datasets:
      meteorological: "ERA5-Land"
      ndvi: "AVHRR_GIMMS"

settings:
  cell_size_km: 1.0
  quality:
    min_time_coverage: 0.95    # 最小时间覆盖率 (95%)
    max_missing_rate: 0.05     # 最大缺失率 (5%)
    max_consecutive_missing: 3  # 最大连续缺失月数
```

**输出目录结构：**

```
example/tuotuohe_1990_2000/
├── drivers/
│   ├── static.nc         # 静态数据
│   ├── dynamic.nc        # 动态驱动数据
│   └── observations.nc   # 观测数据
├── quality_report.json   # 质量报告（--quality-report 选项）
└── metadata.json         # 元数据
```

**质量报告内容：**

质量报告评估以下指标：
- **覆盖率 (Coverage)**: 有效数据比例，阈值 95%
- **缺失率 (Missing Rate)**: 缺失数据比例，阈值 5%
- **连续缺失 (Consecutive Missing)**: 最大连续缺失月数，阈值 3
- **异常值 (Outliers)**: 使用 IQR 方法检测

质量评估通过标准：所有指标均在阈值范围内。

**注意：** 数据准备统一使用 `scripts/prepare_basin_drivers.py`。仓库已不再保留旧版的流域专用和 legacy 数据准备脚本。

### 2.3 流域面积计算

流域面积 (`s_area`) 从 `static.nc` 中的 `BasinMask` 自动计算：

- 每个 grid cell = 1 km²
- s_area (公顷) = 有效网格数 × 100

## 3. 参数率定

### 3.1 基本用法

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --maxiter 100
```

**增加常用参数：**

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河" \
  --run-mode point \
  --selector best_only \
  --popsize 8 \
  --maxiter 100
```

**参数说明：**

| 参数 | 含义 | 默认值 | 必需 | 备注 |
|------|------|--------|------|------|
| `--static-nc` | 静态输入文件路径 | 无 | 是 | `static.nc` |
| `--dynamic-nc` | 动态驱动文件路径 | 无 | 是 | `dynamic.nc` |
| `--observations-nc` | 观测文件路径 | 无 | 是 | `observations.nc` |
| `--station-name` | 站点名称元数据 | `"unknown"` | 否 | 写入输出文件属性 |
| `--model-type` | 率定模型类型 | `crsem` | 否 | 可选 `crsem` 或 `rusle` |
| `--run-mode` | 率定输入模式 | `point` | 否 | `point` 更快，`gridded` 保留空间异质性 |
| `--optimizer` | 优化器名称 | `differential_evolution` | 否 | 可选 `differential_evolution` 或 `glue` |
| `--objective-method` | 目标函数指标 | `nse` | 否 | 传给率定接口的目标函数名称；内置支持 `nse`、`nse_pbias`、`kge`、`kge_pbias`、`rmse`、`mae`、`r2` |
| `--config` | 参数配置 YAML 路径 | 无 | 否 | 覆盖内置参数默认值、边界和惩罚 |
| `--selector` | 率定后成员选择方式 | 自动 | 否 | `differential_evolution` 缺省为 `aic`，`glue` 缺省为 `glue` |
| `--aic-numbers` | 固定返回集合成员数 | 无 | 否 | 仅在 `--selector aic` 时有效 |
| `--aic-max-numbers` | AIC 自动选集时的最大成员数 | 无 | 否 | 仅在 `--selector aic` 时有效 |
| `--aic-delta-threshold` | AIC 差值阈值 | 无 | 否 | 仅在 `--selector aic` 时有效 |
| `--aic-cum-weight` | AIC 累积权重阈值 | 无 | 否 | 仅在 `--selector aic` 时有效 |
| `--maxiter` | 最大迭代次数 | `100` | 否 | DE 时是迭代上限；GLUE 时可作为样本数回退值 |
| `--popsize` | 差分进化种群倍数 | 优化器默认值 | 否 | 显式传入时覆盖优化器默认设置 |
| `--workers` | 并行 worker 数 | 串行 | 否 | 仅 `gridded` 模式支持，`-1` 表示使用全部 CPU |
| `--polish` | 开启末端局部搜索 | `False` | 否 | 启用 L-BFGS-B 精修，耗时更高 |
| `--n-samples` | GLUE 样本数 | 无 | 否 | 仅在 `--optimizer glue` 时有效；未传时回退到 `--maxiter` |
| `--sampling-method` | GLUE 采样方式 | `sobol` | 否 | 可选 `sobol`、`lhs`、`random` |
| `--seed` | 随机种子 | 优化器默认值 | 否 | 控制差分进化和 scrambled GLUE 采样的可复现性 |
| `--glue-threshold` | GLUE 行为参数阈值 | 无 | 否 | 对 NSE/KGE/R² 是下限；对 RMSE/MAE 是上限 |
| `--glue-top-fraction` | GLUE 备选保留比例 | 无 | 否 | 当未提供阈值且无默认阈值时使用 |
| `--glue-max-members` | GLUE 最大返回成员数 | 无 | 否 | 控制 behavioral set 大小 |
| `--glue-channel-ratio-lower` | GLUE 渠道贡献比下限 | 无 | 否 | `channel_ratio = 渠道净贡献 / 坡面来沙` |
| `--glue-channel-ratio-upper` | GLUE 渠道贡献比上限 | 无 | 否 | `channel_ratio = 渠道净贡献 / 坡面来沙` |
| `--save [PATH]` | 保存率定结果 | 不保存 | 否 | 裸 `--save` 输出到数据目录；`PATH` 可指定目录或 `.json` 文件 |
| `--plot-progress` | 绘制率定过程图 | `False` | 否 | 仅 `point` 模式支持 |

**运行模式选择：**

| 参数 | 说明 | 适用场景 |
|------|------|----------|
| `--run-mode point` | 空间平均后的点数据（默认） | 快速率定；支持绘图和非绘图 |
| `--run-mode gridded` | 保留空间异质性的网格数据 | 精细率定；支持 `--workers`，不支持绘图 |

**运行时间对比：**

| 模式 | 网格规模 | 时间维度 | 平均运行时间 |
|------|----------|----------|-------------|
| Grid | 175×256 (23,354 像元) | 132 月 | ~415 ms |
| Point | 1 (空间平均) | 132 月 | ~2.6 ms |

Grid 模式比 Point 模式慢约 **160 倍**。对于典型校准场景（如 `maxiter=40`），Point 模式仅需几秒，Grid 模式需要几分钟到几十分钟。

**输出详细程度控制：**

```bash
# 默认模式：不绘图，自动使用精简诊断输出
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河"

# point 模式可开启实时绘图
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河" \
  --run-mode point \
  --plot-progress

# gridded 模式不支持绘图
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --run-mode gridded
```

| 模式 | 内部诊断输出 | 适用场景 |
|------|----------|----------|
| 默认（不加 `--plot-progress`） | `compact`，仅保留 SSF_pred, A_channel, R_rain, R_melt | 更快的率定 |
| `point + --plot-progress` | `full`，额外保留 K/C/SDR 等诊断变量 | 交互式观察率定过程 |

**限制：** `gridded` 率定不支持 `--plot-progress`。

**说明：** 率定结果默认保存到 `static.nc` 所在目录。
进度图左下角中，`Loss` 表示当前目标函数值，`Conv` 表示优化器收敛指标；二者不是同一个量。

### 3.2 使用配置文件管理参数

参数率定支持使用 YAML 配置文件来管理参数的初始值、边界和惩罚设置。如果不提供配置文件，将使用代码内置的默认值。

**配置文件格式：**

```yaml
# config/parameter_config.custom.yml
model_type: crsem

defaults:
  a_rain: 0.5
  r_th: 1
  a_melt: 0.1
  # ... 其他参数

bounds:
  a_rain: [0.5, 1.0]
  r_th: [1, 20]
  # ... 其他参数边界

penalties:
  channel_ratio:
    enabled: true
    lower_bound: -0.6
    upper_bound: 0.3
  annual_r_factor:
    enabled: true
    lower_normal: 100.0
    upper_normal: 200.0
```

其中 `channel_ratio = 渠道净贡献 / 坡面来沙`。负值表示净沉积，正值表示净侵蚀。

**使用配置文件进行率定：**

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河" \
  --config config/parameter_config.custom.yml \
  --maxiter 100
```

**项目内置的配置文件：**

- `config/parameter_config.crsem.yml`: CRSEM 模型参数配置
- `config/parameter_config.rusle.yml`: RUSLE 模型参数配置

### 3.3 集合率定模式

**AIC 自动选择集合成员：**

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河" \
  --selector aic \
  --aic-max-numbers 5 \
  --aic-delta-threshold 6 \
  --maxiter 100
```

**固定集合成员数：**

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --station-name "沱沱河" \
  --selector aic \
  --aic-numbers 3 \
  --maxiter 100
```
### 3.4 率定输出

**控制台输出：**

率定完成后，控制台会输出格式化的结果摘要，包括：

1. **优化结果**：优化器信息、成功状态、最终目标函数值
2. **性能指标**：NSE、振幅比、河道侵蚀比
3. **参数表格**：包含初始值、率定值、参数边界和边界接近指示器

**参数表格说明：**

```
Parameter             Initial     Calibrated                 Bounds      →
--------------------------------------------------------------------------------
a_rain               0.500000       0.572440       [0.5000, 1.0000]
r_th                 1.000000      19.765462      [1.0000, 20.0000]      ↑
alpha_C              2.000000       4.790071       [1.0000, 5.0000]      ↑
k_melt               3.000000       1.133976       [1.0000, 5.0000]      ↓
...

Legend: ↓ = near lower bound (within 5% of range)
        ↑ = near upper bound (within 5% of range)
```

**边界接近指示器：**
- `↑`：参数值接近上边界（在范围的后 10% 内）
- `↓`：参数值接近下边界（在范围的前 10% 内）
- 无标识：参数值在边界范围内

**注意：** 如果率定参数频繁触碰边界，可能表示：
- 参数边界设置过窄
- 模型结构需要调整
- 观测数据存在问题

**输出文件：**

```
example/tuotuohe_1990_2000/
└── params.json    # 率定参数文件
```

**params.json 内容：**

```json
{
  "parameter_batch": {
    "param_names": ["a_rain", "r_th", ...],
    "values": [[0.57, 18.47, ...], ...],
    "weights": [0.53, 0.32, 0.15]
  },
  "metrics": {
    "NSE": 0.83,
    "KGE": 0.82,
    "R2": 0.84,
    "dynamic": "example/tuotuohe_1990_2000/drivers/dynamic.nc",
    "static": "example/tuotuohe_1990_2000/drivers/static.nc",
    "observations": "example/tuotuohe_1990_2000/drivers/observations.nc",
    ...
  }
}
```

**率定指标：**

| 指标 | 说明 |
|------|------|
| NSE | Nash-Sutcliffe 效率系数 |
| KGE | Kling-Gupta 效率系数 |
| R² | 决定系数 |
| RMSE | 均方根误差 |
| MAE | 平均绝对误差 |

**率定效果展示：**

沱沱河流域 1990-2000 年率定结果示例：

![SSF 对比图](ssf_comparison.png)

图中展示了模拟输沙量与观测值的对比，包括时间序列、散点图、月均气候态和年总量。

## 4. 模型运行

### 4.1 点模式运行

**最小命令：**

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/params.json \
  --start-year 1990 \
  --end-year 2000 \
  --output-file example/tuotuohe_1990_2000
```

**时间序列输沙模拟（推荐用于率定验证）：**

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/params.json \
  --start-year 1990 \
  --end-year 2000 \
  --run-mode point \
  --run-method run_hillslope_river
```

**参数说明：**

| 参数 | 含义 | 默认值 | 必需 | 备注 |
|------|------|--------|------|------|
| `--static-nc` | 静态输入文件路径 | 无 | 是 | `static.nc` |
| `--dynamic-nc` | 动态驱动文件路径 | 无 | 是 | `dynamic.nc` |
| `--observations-nc` | 观测文件路径 | 无 | 是 | `observations.nc` |
| `--station-name` | 站点名称元数据 | `"unknown"` | 否 | 写入输出文件属性 |
| `--start-year` | 模拟开始年份 | 驱动起始年份 | 否 | 对输入驱动做时间裁剪（含起始年） |
| `--end-year` | 模拟结束年份 | 驱动结束年份 | 否 | 对输入驱动做时间裁剪（含结束年） |
| `--params-file` | 率定参数文件路径 | 无 | 是 | `params.json` |
| `--run-method` | 模型执行入口 | `run_hillslope_river` | 否 | 可选 `run_hillslope` 或 `run_hillslope_river` |
| `--run-mode` | 运行输入模式 | `gridded` | 否 | `point` 用空间平均输入，`gridded` 用原始网格 |
| `--aggregate` | 集合成员聚合方式 | `none` | 否 | `none` 保留所有参数成员，其他值由聚合器解释 |
| `--output-file [PATH]` | 输出 NetCDF 路径 | 不保存 | 否 | 裸 `--output-file` 输出到默认 `model_output.nc`；传目录时写到 `PATH/model_output.nc`；传 `.nc` 路径时写到该文件 |

**说明：** 运行模式默认是 `gridded`；模型类型从 `params.json` 自动识别。添加 `--start-year/--end-year` 可在运行前裁剪模拟时间窗。添加 `--output-file` 后会把输出写到数据目录，默认文件名为 `model_output.nc`。如果 `dynamic.nc` 的 `NDVI` 含有 `member` 维，标准运行会先对 NDVI 成员求均值，再执行模型；这里的 `--aggregate` 只针对参数集合输出，不处理 NDVI 集合展开。

**输出变量：**

- `SSF_pred`: 预测输沙量 (吨/月)
- `A_channel`: 河道侵蚀/沉积贡献 (吨/月)
- `E_hillslope`: 坡面输沙量 (吨/月)

**维度：** `(member, time)`

### 4.2 空间模式运行

**空间分布模拟（推荐用于空间分析）：**

```bash
python scripts/run_model.py \
  --static-nc example/tuotuohe_1990_2000/drivers/static.nc \
  --dynamic-nc example/tuotuohe_1990_2000/drivers/dynamic.nc \
  --observations-nc example/tuotuohe_1990_2000/drivers/observations.nc \
  --params-file example/tuotuohe_1990_2000/params.json \
  --run-mode gridded \
  --run-method run_hillslope_river
```

**输出变量：**

| 变量 | 说明 | 维度 |
|------|------|------|
| SSF_pred | 预测输沙量 | (member, time) |
| A_channel | 河道贡献 | (member, time) |
| E_hillslope | 坡面输沙 | (member, time) |
| R_rain | 降雨侵蚀力 | (member, time, y, x) |
| R_melt | 融雪侵蚀力 | (member, time, y, x) |
| K_factor | 土壤可蚀性因子 | (member, time, y, x) |
| C_factor | 植被覆盖因子 | (member, time, y, x) |
| SDR | 泥沙输移比 | (member, time, y, x) |

### 4.3 集合结果聚合

**加权平均：**

```bash
python scripts/run_model.py \
  ... \
  --aggregate weighted_mean
```

**保留所有成员：**

```bash
python scripts/run_model.py \
  ... \
  --aggregate none
```

**说明：** 使用 `--output-file` 可写出默认输出文件；传目录时自动写成 `PATH/model_output.nc`；传 `.nc` 路径时写到指定文件。

### 4.4 模型输出属性

`model_output.nc` 包含以下全局属性，记录率定信息：

| 属性 | 说明 |
|------|------|
| calibration_start_year | 率定开始年份 |
| calibration_end_year | 率定结束年份 |
| calibration_NSE | 率定期 NSE |
| calibration_KGE | 率定期 KGE |
| calibration_R2 | 率定期 R² |
| station_name | 站点名称 |

这些属性用于绘图脚本自动识别率定时段。

## 5. 结果可视化

### 5.1 SSF 对比绘图

使用 `scripts/plot_ssf_comparison.py` 绘制模拟与观测输沙量对比图：

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000/model_output.nc \
  --observed example/tuotuohe_1990_2000/drivers/observations.nc \
  --output example/tuotuohe_1990_2000/ssf_comparison.png
```

**参数说明：**

| 参数 | 含义 | 默认值 | 必需 | 备注 |
|------|------|--------|------|------|
| `--simulated` | 模拟结果 NetCDF 路径 | 无 | 是 | 通常为 `model_output.nc` |
| `--observed` | 观测结果 NetCDF 路径 | 无 | 是 | 通常为 `observations.nc` |
| `--calibration-start` | 率定期开始年份 | 从模拟结果属性读取 | 否 | 缺省时尝试读取 NetCDF 属性 |
| `--calibration-end` | 率定期结束年份 | 从模拟结果属性读取 | 否 | 缺省时尝试读取 NetCDF 属性 |
| `--output` | 图像输出路径 | 模拟结果同目录下的 `ssf_comparison.png` | 否 | 可写成任意 PNG 路径 |
| `--title` | 图标题 | 自动生成 | 否 | 为空时根据数据自动生成 |
| `--member` | 集合结果选择方式 | `auto` | 否 | 单成员文件会忽略该参数；集合文件支持 `auto`/`mean`、零基索引或成员标签 |
| `--force-split` | 强制划分率定/验证期 | `False` | 否 | 即使模拟期与率定期一致也强制拆分 |

**绘图内容：**

- 时间序列对比图（含率定期阴影标注）
- 散点图（含 1:1 线和 R²）
- 月均气候态对比
- 年总量对比

**自动时期分割：**

脚本会自动判断是否需要区分率定期和验证期：

| 条件 | 行为 |
|------|------|
| 模拟时段 = 率定时段 | 不分割，显示单一指标 |
| 模拟时段 > 率定时段 | 自动分割，分别计算率定期和验证期指标 |

**手动指定率定期：**

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000_1985_2015/model_output.nc \
  --observed example/tuotuohe_1990_2000_1985_2015/observations.nc \
  --calibration-start 1990 \
  --calibration-end 2000 \
  --output figures/ssf_comparison.png
```

**强制分割：**

```bash
python scripts/plot_ssf_comparison.py \
  --simulated example/tuotuohe_1990_2000/model_output.nc \
  --observed example/tuotuohe_1990_2000/observations.nc \
  --calibration-start 1990 \
  --calibration-end 1995 \
  --force-split \
  --output figures/ssf_comparison_split.png
```

**输出指标：**

| 指标 | 说明 |
|------|------|
| NSE | Nash-Sutcliffe 效率系数 |
| KGE | Kling-Gupta 效率系数 |
| PBIAS | 百分比偏差 |
| R² | 决定系数 |
| n | 数据点数 |

## 6. 驱动变量归因分析

使用 [`scripts/attribution_analysis.py`](/mnt/d/code/sediment/scripts/attribution_analysis.py) 可以定量分析 `NDVI`、`T` 或 `Pre` 的变化对流域坡面侵蚀入河量变化的贡献。该脚本与 [`CRSEM/sensitivity.py`](/mnt/d/code/sediment/CRSEM/sensitivity.py) 不同，后者回答的是“敏感性和相对重要性”，而不是“累计贡献量”。

**方法说明：**

- 真实情景：使用原始驱动序列运行模型
- 反事实情景：将目标变量替换为基准期气候态季节循环
- 贡献定义：`ΔE = E_real - E_counterfactual`
- 模型入口：固定使用 `run_hillslope`，不考虑 `Q` 和河道输移过程

对于 `NDVI` 集合驱动，归因脚本会逐个 `ndvi_member` 运行，再合并为统一输出。这和标准率定/标准模拟不同：后两者会先把 NDVI 成员折叠为均值场，再进入模型。

```bash
python scripts/attribution_analysis.py \
  --static example/tuotuohe/drivers/static.nc \
  --dynamic example/tuotuohe/drivers/dynamic.nc \
  --params example/tuotuohe_1990_2000/params.json \
  --variable NDVI \
  --analysis-start 1987 \
  --analysis-end 2022 \
  --baseline-start 1987 \
  --baseline-end 2000 \
  --output-dir example/tuotuohe/attribution
```

**参数说明：**

| 参数 | 含义 |
|------|------|
| `--static` | 驱动目录中的 `static.nc` |
| `--dynamic` | 驱动目录中的 `dynamic.nc` |
| `--observations` | 可选的 `observations.nc`；当前归因流程不会使用其中的 `Q` 或 `SSF` |
| `--params` | 率定得到的参数集合 `params.json` |
| `--variable` | 要归因的变量，支持 `NDVI`、`T`、`Pre` |
| `--analysis-start` / `--analysis-end` | 归因模拟时间范围；默认使用 `dynamic.nc` 的完整时间范围 |
| `--baseline-start` / `--baseline-end` | 反事实构造所用基准期 |
| `--output-dir` | 结果输出目录 |
| `--no-point-mode` | 保持网格维而非转换到流域平均点模式 |

**输出变量：**

| 变量 | 含义 | 单位 |
|------|------|------|
| `delta_annual` | 年尺度贡献；若输出包含 `E_hillslope`，默认表示坡面侵蚀模数变化 | `t ha-1 yr-1` |
| `delta_cumulative` | 将 `delta_annual` 乘以流域面积换算到总量后累计求和 | `t` |

**维度说明：**

- `time`: 年尺度时间轴
- `member`: 参数集合成员
- `ndvi_member`: NDVI 集合成员，仅在 `NDVI` 为多成员输入时存在

**结果解读：**

- `delta_cumulative < 0` 表示相对于基准期气候态，变量变化带来了累计减蚀效应
- `delta_cumulative > 0` 表示累计增蚀效应
- 如果需要单一流域结论，通常应先对 `member` 做参数权重平均，再比较不同 `ndvi_member`

仓库已附带一个沱沱河示例结果文件：

- [`example/tuotuohe/attribution/attribution_NDVI_1987_2000.nc`](/mnt/d/code/sediment/example/tuotuohe/attribution/attribution_NDVI_1987_2000.nc)

归因结果可继续使用 [`scripts/plot_ndvi_attribution_analysis.py`](/mnt/d/code/sediment/scripts/plot_ndvi_attribution_analysis.py) 生成三联图：

```bash
python scripts/plot_ndvi_attribution_analysis.py \
  --attribution-nc example/tuotuohe/attribution/attribution_NDVI_1987_2000.nc \
  --dynamic-nc example/tuotuohe/drivers/dynamic.nc \
  --output example/tuotuohe/attribution/attribution_NDVI_1987_2000.png
```

## 7. 运行模式对比

| run_method | run_mode | 适用场景 | 输出维度 |
|------------|----------|----------|----------|
| run_hillslope | point | 坡面过程分析 | (member, time) |
| run_hillslope | gridded | 空间侵蚀分布 | (member, time, y, x) |
| run_hillslope_river | point | 率定验证、输沙预测 | (member, time) |
| run_hillslope_river | gridded | 综合分析 | `SSF_pred` / `A_channel` 为 `(member, time)`；坡面变量为 `(member, time, y, x)` |

**推荐用法：**

- 参数率定：使用 `run_hillslope_river + point`
- 输沙预测：使用 `run_hillslope_river + point`
- 空间分析：使用 `run_hillslope + gridded`

## 8. Python API 使用

### 8.1 直接使用 BasinDriver

```python
from CRSEM.driver import BasinDriver

# 加载数据
driver = BasinDriver.from_nc_files(
    static_nc="example/tuotuohe_1990_2000/static.nc",
    dynamic_nc="example/tuotuohe_1990_2000/dynamic.nc",
    observations_nc="example/tuotuohe_1990_2000/observations.nc",
    station_name="沱沱河"
)

# 获取运行上下文
ctx = driver.to_run_context()

# 访问数据
print(driver.s_area)  # 流域面积 (公顷)
print(driver.Q)       # 流量序列
print(driver.SSF)     # 输沙量序列
```

### 8.2 运行模型

```python
from CRSEM.batch_runner import run_parameter_batch
from CRSEM.contracts import ParameterBatch

# 加载率定参数
params, metrics = ParameterBatch.from_file("calibration_results/params.json")

# 点模式运行
point_driver = driver.to_point_driver(keep_rivers=True)
result = run_parameter_batch(
    model_type="crsem",
    source=point_driver,
    params=params,
    run_method="run_hillslope_river"
)

# 获取结果
ds = result.to_dataset()
ssf_pred = ds["SSF_pred"]  # (member, time)
```

## 9. 单位说明

| 变量 | 单位 |
|------|------|
| s_area | 公顷 (ha) |
| SSF | 吨/月 (t/month) |
| Q | 立方米/秒 (m³/s) |
| Pre | 毫米/月 (mm/month) |
| T | 摄氏度 (°C) |
| E_hillslope | 吨/公顷/月 (t/ha/month) |
| R_rain, R_melt | MJ·mm/(ha·h·month) |

## 10. 常见问题

### Q1: 率定 NSE 为负值？

可能原因：
- 初始参数范围不合适
- 迭代次数不足
- 驱动数据与观测数据时间不对齐

解决方案：
- 增加 `--maxiter`
- 检查数据时间范围是否一致

### Q2: 预测 SSF 量级异常？

检查：
- s_area 单位是否正确（应为公顷）
- 观测 SSF 单位是否正确（应为吨/月）
- Pre 单位是否正确（应为 mm/月）

### Q3: 空间模式内存不足？

解决方案：
- 使用较小的时空范围
- 分段运行
- 增加系统内存
