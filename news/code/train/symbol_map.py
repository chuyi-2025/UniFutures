#!/usr/bin/env python3
"""Map Chinese futures names from OCR headers to UniFutures contract symbols."""

from __future__ import annotations

# Longer aliases first when matching substrings.
SYMBOL_ALIASES: dict[str, str] = {
    # metals
    "不锈钢": "SS",
    "氧化铝": "AO",
    "工业硅": "SI",
    "多晶硅": "PS",
    "碳酸锂": "LC",
    "白银": "AG",
    "沪银": "AG",
    "黄金": "AU",
    "沪金": "AU",
    "沪铜": "CU",
    "铜": "CU",
    "沪铝": "AL",
    "铝": "AL",
    "沪锌": "ZN",
    "锌": "ZN",
    "沪铅": "PB",
    "铅": "PB",
    "沪镍": "NI",
    "镍": "NI",
    "沪锡": "SN",
    "锡": "SN",
    "国际铜": "BC",
    # ferrous
    "铁矿石": "I",
    "螺纹钢": "RB",
    "热轧卷板": "HC",
    "热卷": "HC",
    "焦煤": "JM",
    "焦炭": "J",
    "玻璃": "FG",
    "硅铁": "SF",
    "锰硅": "SM",
    "硅锰": "SM",
    "钢材": "RB",
    # energy/chem
    "原油": "SC",
    "低硫燃料油": "LU",
    "燃料油": "FU",
    "沥青": "BU",
    "石油沥青": "BU",
    "甲醇": "MA",
    "PTA": "TA",
    "pta": "TA",
    "精对苯二甲酸": "TA",
    "对二甲苯": "PX",
    "PX": "PX",
    "乙二醇": "EG",
    "MEG": "EG",
    "苯乙烯": "EB",
    "纯苯": "BZ",
    "尿素": "UR",
    "纯碱": "SA",
    "烧碱": "SH",
    "PVC": "V",
    "聚氯乙烯": "V",
    "聚丙烯": "PP",
    "PP": "PP",
    "塑料": "L",
    "聚乙烯": "L",
    "LLDPE": "L",
    "短纤": "PF",
    "瓶片": "PR",
    "瓶级切片": "PR",
    "LPG": "PG",
    "液化石油气": "PG",
    "合成橡胶": "BR",
    "丁二烯橡胶": "BR",
    "丁二烯": "BR",
    "20号胶": "NR",
    "20号橡胶": "NR",
    "天然橡胶": "RU",
    "橡胶": "RU",
    "纸浆": "SP",
    "原木": "LG",
    "集运指数（欧线）": "EC",
    "集运欧线": "EC",
    "集运": "EC",
    # ag
    "豆粕": "M",
    "菜粕": "RM",
    "菜籽粕": "RM",
    "豆油": "Y",
    "棕榈油": "P",
    "菜油": "OI",
    "菜籽油": "OI",
    "豆一": "A",
    "大豆": "A",
    "豆二": "B",
    "黄大豆1号": "A",
    "黄大豆2号": "B",
    "玉米淀粉": "CS",
    "淀粉": "CS",
    "玉米": "C",
    "棉花": "CF",
    "棉纱": "CY",
    "白糖": "SR",
    "食糖": "SR",
    "红枣": "CJ",
    "苹果": "AP",
    "花生": "PK",
    "生猪": "LH",
    "鸡蛋": "JD",
    "油脂": "Y",
    # financial
    "中证500": "IC",
    "中证1000": "IM",
    "沪深300": "IF",
    "上证50": "IH",
    "股指期货": "IF",
    "股指": "IF",
    "二年期国债": "TS",
    "五年期国债": "TF",
    "十年期国债": "T",
    "三十年期国债": "TL",
    "国债期货": "T",
    "国债": "T",
}


def normalize_name(name: str) -> str:
    s = name.strip()
    for ch in ("（", "）", "(", ")", " ", "\u3000", "期货", "主力", "合约"):
        s = s.replace(ch, "")
    return s


def map_variety(name: str) -> str | None:
    """Map one Chinese variety token to a contract symbol, or None."""
    raw = name.strip()
    if not raw or raw in {"未知", "多品种", "大宗商品", "有色", "黑色", "能化", "农产品"}:
        return None

    key = normalize_name(raw)
    # Prefer explicit aliases first so tokens like PTA/PX map correctly.
    if key in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[key]
    if raw in SYMBOL_ALIASES:
        return SYMBOL_ALIASES[raw]
    upper = raw.upper()
    known_codes = set(SYMBOL_ALIASES.values())
    if upper in known_codes:
        return upper

    # longest alias substring match
    best: tuple[int, str] | None = None
    for alias, sym in SYMBOL_ALIASES.items():
        if alias in key or key in alias:
            score = len(alias)
            if best is None or score > best[0]:
                best = (score, sym)
    return best[1] if best else None


def map_varieties(field: str) -> list[str]:
    """Split header 品种 field and map to unique symbols (order preserved)."""
    parts = []
    for tok in field.replace("／", "/").replace("、", "/").replace("，", "/").replace(",", "/").split("/"):
        sym = map_variety(tok)
        if sym and sym not in parts:
            parts.append(sym)
    return parts
