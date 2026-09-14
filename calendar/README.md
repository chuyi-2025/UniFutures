# Calendar / Roll / Delivery-Month Strategies

非天气的 **时间周期性** 策略：换月窗口、交割月、日历月。

| 文档 | 说明 |
|------|------|
| [docs/CALENDAR_ROLL_STRATEGY.md](docs/CALENDAR_ROLL_STRATEGY.md) | 策略定义、回测摘要、进组合建议 |

```bash
python3 calendar/code/build_calendar_panels.py --start 2014-01-01
python3 calendar/code/search_calendar_strategies.py --start 2018-01-01
```

与 `spread/` 的区别：`spread` 做全样本 carry MR；`calendar` 强调 **事件门控**（换月/交割/日历月）。
