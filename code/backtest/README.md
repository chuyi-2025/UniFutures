# SN 模型回测对比

品种：**SN（锡）**  |  区间：2024-01-01 ~ 2026-12-31  |  初始资金：**100 万**

## 统一回测设定

| 项目 | 设定 |
|------|------|
| 合约选择 | 每日主力：交割月 > 当前月 + 3 的最近合约 |
| 持仓 | 单品种单合约，满仓（±1），初始 100 万 |
| 收益计算 | `capital *= exp(position × log_ret_1d)` |
| 调仓周期 | Kronos：**日频**；GAF：**3 日**（与 ret_3d 一致）；XGB：**5 日**（与 ret_5d 一致） |
| 换向规则 | Kronos 日频：反向当日只平不开；GAF/XGB 持仓期内不调仓 |

## 模型方案差异

| 模型 | 输入 | 训练目标 | 信号规则 | 训练范围 |
|------|------|----------|----------|----------|
| **XGB** | 75 维工程特征 + 合约编码 | 5 日收益 8 分类 | class 0→空，class 7→多，其余观望；**5 日调仓** | 全品种（剔除 WR/ZC 等 10 个） |
| **Kronos** | 64×6 OHLCV 归一化窗口 | 预训练 TS 模型，预测次日 close | pred_ret > 0.001 多，< -0.001 空；**日频** | 预训练权重，零样本推理 |
| **gaf_cnn_ferrous** | 64×64 GAF(close) | 3 日收益 MSE（±5% clip） | pred_ret > 0.001 多，< -0.001 空；**3 日调仓** | 黑色系品种 |
| **gaf_cnn_nonferrous** | 同上 | 同上 | 同上 | 有色金属（含 SN） |
| **gaf_cnn_oilseeds** | 同上 | 同上 | 同上 | 油脂油料 |

> GAF / Kronos 阈值 `TH=0.001`。XGB 为离散分类信号。
> XGB 特征在部分主力日缺失时，调仓日 signal=0 则平仓观望。

## SN 回测结果

| 排名 | 模型 | 期末资金 (万) | 收益率 | Sharpe | 最大回撤 | 持仓天数 |
|------|------|--------------|--------|--------|----------|----------|
| 1 | gaf_cnn_nonferrous | 247.6 | 147.57% | 1.31 | -21.51% | 523 |
| 2 | Kronos | 158.2 | 58.19% | 0.64 | -33.22% | 572 |
| 3 | gaf_cnn_oilseeds | 148.1 | 48.05% | 0.66 | -20.41% | 399 |
| 4 | XGB | 109.4 | 9.45% | 0.15 | -30.20% | 235 |
| 5 | gaf_cnn_ferrous | 85.0 | -14.99% | -0.28 | -29.17% | 318 |

## 复现命令

```bash
cd /home/workspace/lab/UniFutures/code/backtest
python3 compare_sn_models.py
```

或单独运行：

```bash
python3 backtest_xgb.py --symbol SN --out-dir ../../data/results/compare_sn/xgb
python3 backtest_kronos.py --symbol SN --out-dir ../../data/results/compare_sn/kronos
python3 backtest_gaf.py --symbol SN --model-path ../../data/models/gaf_cnn_nonferrous.pt \
  --out-dir ../../data/results/compare_sn/gaf_cnn_nonferrous
```

详细日频结果：`data/results/compare_sn/<model>/daily.csv`
