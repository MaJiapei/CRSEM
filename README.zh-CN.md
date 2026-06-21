# CRSEM

CRSEM 是一个面向寒区流域的土壤侵蚀与输沙建模工作流，使用月尺度驱动数据，支持已制备 NetCDF 输入、参数率定、流域模拟和植被变化归因分析。

英文说明见 [README.md](README.md)。详细文档见 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)、[docs/USER_GUIDE.en.md](docs/USER_GUIDE.en.md)、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 和 [docs/ARCHITECTURE.en.md](docs/ARCHITECTURE.en.md)。

## 仓库内容

- Python 包：[CRSEM](CRSEM)
- 命令行工作流：[scripts](scripts)
- 测试：[tests](tests)
- 配置示例：[config](config)
- 内置轻量真实案例：[example/zhimenda_sample](example/zhimenda_sample)

## 内置直门达案例

仓库包含一个直门达流域轻量案例。该案例由完整网格驱动数据提取为 1x1 的流域平均样例，因此可以直接放在 GitHub 中，同时保留真实直门达驱动、观测和率定参数。

- [static.nc](example/zhimenda_sample/drivers/static.nc)
- [dynamic.nc](example/zhimenda_sample/drivers/dynamic.nc)
- [observations.nc](example/zhimenda_sample/drivers/observations.nc)
- [params_1982_2000_kge_pbias_m120.json](example/zhimenda_sample/params_1982_2000_kge_pbias_m120.json)

完整网格版直门达 `dynamic.nc` 约 1.2 GB，不纳入 Git。大型网格驱动、DEM 产品、GIS 中间文件和模型输出 NetCDF 应通过 release asset 或数据仓库存放，而不是直接提交到代码仓库。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

可选依赖：

- `rioxarray` 仅在数据制备阶段需要坐标重投影时使用。

## 快速开始

运行内置直门达案例：

```bash
python scripts/run_model.py \
  --static-nc example/zhimenda_sample/drivers/static.nc \
  --dynamic-nc example/zhimenda_sample/drivers/dynamic.nc \
  --observations-nc example/zhimenda_sample/drivers/observations.nc \
  --params-file example/zhimenda_sample/params_1982_2000_kge_pbias_m120.json \
  --station-name zhimenda \
  --run-mode point \
  --start-year 1982 \
  --end-year 2019
```

如果要保存输出，可追加：

```bash
--output-file example/zhimenda_sample/output/model_output.nc
```

输出文件默认被 Git 忽略。

基于同一案例做一次轻量率定：

```bash
python scripts/calibrate_parameters.py \
  --static-nc example/zhimenda_sample/drivers/static.nc \
  --dynamic-nc example/zhimenda_sample/drivers/dynamic.nc \
  --observations-nc example/zhimenda_sample/drivers/observations.nc \
  --station-name zhimenda \
  --calibration-start 1982 \
  --calibration-end 2000 \
  --maxiter 3
```

率定默认使用 `point` 模式。如果切换到 `--run-mode gridded`，则不支持进度绘图，并可使用 `--workers`。

## 测试

```bash
python -m pytest tests -q
```

如果只想快速确认内置真实案例可运行：

```bash
python -m pytest tests/test_real_data_smoke.py -q
```

## 发布前仍需确认

- 项目许可证
- 内置数据是否具备公开再分发权限
- 是否需要 release tag 和引用元数据

如果没有 `LICENSE` 文件，其他用户虽然能看到代码，但没有明确的复用授权。
