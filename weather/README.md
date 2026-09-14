# 国内天气 × 农产品策略

用中国主产区天气驱动国内定价主导的农产品研究信号。第一版数据来自
[Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)
（按国内产区经纬度取日值，免费无钥）。接口形态预留日后接 CMA / 国内站点爬虫。

**方案全文**：[docs/WEATHER_AG_STRATEGY.md](./docs/WEATHER_AG_STRATEGY.md)

## 品种（弱外盘）

| 代码 | 品种 | 主产区代表点 |
|------|------|--------------|
| CF | 棉花 | 石河子 / 阿克苏 / 库尔勒 |
| SR | 白糖 | 崇左 / 南宁 / 临沧 |
| AP | 苹果 | 洛川 / 栖霞 |
| CJ | 红枣 | 若羌 / 阿克苏 |

不做大豆、豆粕、棕榈等外盘主导品种；玉米暂不做。

## 目录

```
weather/
  config/crop_regions.yaml
  code/          # fetch → factors → align → backtest
  data/raw/      # 站点日值 parquet
  data/factors/  # 品种日频因子
  data/results/  # 研究回测输出
```

## 跑通

```bash
cd /home/workspace/lab/UniFutures

# 1) 拉历史日值（默认同配置 start_date→今天；可 --start/--end）
python3 weather/code/fetch_open_meteo.py

# 2) 产区加权 + 气候态异常因子
python3 weather/code/build_factors.py

# 3) 对齐期货收益并研究回测（朴素应激规则）
python3 weather/code/backtest_weather.py

# 4) 多方案搜索（Sharpe / IC / IR + 时间对半）
python3 weather/code/search_weather_strategies.py
python3 weather/code/search_weather_phase2.py
```

研究回测为示意口径（单品种 ±1 手示意收益），非正式账户 blotter。

## 搜索结论（2018+，研究口径）

产出目录：`weather/data/results/search/`（`phase2_summary.json`、`robust_schemes.csv`、`picked_book_daily.csv`）。

| 层级 | 结果 |
|------|------|
| 朴素应激→多 | Sharpe 为负，不可用 |
| 等权书网格（phase1） | 最高 Sharpe≈0.83，但 IC≈0（稀疏信号噪音） |
| 分品种 + OOS（phase2） | **robust 21 / soft 132**；组合 book Sharpe≈**1.41**，前后半都为正 |

相对更扎实的分品种规则（同时看 Sharpe、5 日 IC/IR、时间对半）：

| 品种 | 因子 | 规则要点 | Sharpe | IC5 / IR5 |
|------|------|----------|--------|-----------|
| CF | `tmean_z_ma5` | 物候窗内，偏低气温→做空（dir=-1, sign, hold5） | 0.68 | 0.14 / 0.25 |
| SR | `tmean_z_ma10` | 偏高气温→做多（sign thr1.5 hold10） | 0.72 | 0.13 / 0.33 |
| AP | `tmean_z_ma5` | 物候窗高温 long_only thr1.5 hold10 | 0.77 | 0.08 / 0.09 |
| CJ | `tmean_z_ma10` | 偏高气温 sign thr1.5 hold5 | 0.87 | 0.02 / 0.12 |

因子纯预测力最强的一档（不交易）：**SR `tmean_z_ma10`** 对 5–10 日收益 IC≈0.13–0.19、IR≈0.33–0.45。

说明：这是 in-sample 网格优选 + 时间对半，不是严格 walk-forward；尚未做账户保证金/手续费 blotter。

## 接 CMA

将 `fetch_open_meteo.py` 换成 CMA API 客户端即可，只要把日值写成同 schema：

`date, tmin, tmax, tmean, precip, rh, station_id`
