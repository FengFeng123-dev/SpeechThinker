import csv
import requests
from langchain_core.tools import tool
from config.settings import settings


def _load_cities():
    path = settings.DATA_DIR / "China-City-List-latest.csv"
    cities = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 14 and row[0] != "Location_ID":
                    cities[row[2].strip()] = row[13].strip()
    return cities


CITY_CODE = _load_cities()


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实况天气。支持全国3500+市县区。
    例如，如问的是广西柳州，则传进去柳州，中国广西贵港，则是贵港，北京，则传进去是北京，其它以此类推，只保留城市名称
    Args:
        city: 城市名称，如"北京"、"深圳"、"海淀"
    """
    key = settings.AMAP_API_KEY
    c = city.strip()
    code = CITY_CODE.get(c)
    if not code:
        matches = [k for k in CITY_CODE if c in k or k.startswith(c) or k in c]
        matches = sorted(matches, key=len)
        if len(matches) == 1:
            code = CITY_CODE[matches[0]]
            c = matches[0]
        elif len(matches) > 1:
            return f"找到多个匹配: {', '.join(matches[:8])}"
    if not code:
        return f"未找到「{city}」，已加载{len(CITY_CODE)}个城市"
    try:
        r = requests.get("https://restapi.amap.com/v3/weather/weatherInfo",
                         params={"key": key, "city": code, "extensions": "base"}, timeout=10)
        d = r.json()
        if d.get("status") != "1":
            return f"❌ API错误({d.get('infocode')}): {d.get('info')}"
        w = d["lives"][0]
        return (
f"""{w['province']} {w['city']}
时间: {w['reporttime']}
天气: {w['weather']}
气温: {w['temperature']}°C
风向风力: {w['winddirection']}风 {w['windpower']}级
湿度: {w['humidity']}%"""
        )
    except requests.Timeout:
        return "❌ 请求超时"
    except Exception as e:
        return f"❌ 错误: {e}"
