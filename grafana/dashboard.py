"""Build editable Grafana panels from the notebook's aggregate data."""
import base64
from getpass import getpass
import json
from pathlib import Path
import os
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


ROOT = "http://localhost:3000"
DASH_UID = "dreambank-dreampoly"
SOURCE = {"uid": "dreampoly-chart-data", "type": "grafana-testdata-datasource"}
WARM = ["#412329", "#6b3033", "#913e37", "#bc593c", "#e6814c"]
COOL = ["#202b48", "#253a67", "#2d5090", "#386ab8", "#4e8be2"]
OR_COLORS = ["#366cb2", "#354c73", "#303747", "#774139", "#bf5544"]
FREQUENCY = "suffix: / 1000"


def override(matcher, value, **properties):
    return {
        "matcher": {"id": matcher, "options": value},
        "properties": [{"id": key, "value": setting} for key, setting in properties.items()],
    }


def panel_options(kind, table, decimals, unit, horizontal=False, colors=None, cuts=()):
    defaults = {"decimals": decimals, "unit": unit, "color": {"mode": "palette-classic"}}
    legend = {"showLegend": len(table["columns"]) > 2, "displayMode": "list", "placement": "bottom"}
    tooltip = {"mode": "multi", "sort": "none"}
    overrides = []
    if kind == "xychart":
        defaults["custom"] = {"show": "points+lines", "lineWidth": 2, "pointSize": {"fixed": 7}}
        overrides.append(override("byName", table["columns"][0], unit="none", decimals=1))
        options = {"seriesMapping": "auto", "legend": legend, "tooltip": {"mode": "single"}}
    elif kind == "barchart":
        options = {
            "xField": table["columns"][0], "orientation": "horizontal" if horizontal else "vertical",
            "stacking": "none", "showValue": "never", "fillOpacity": 82, "groupWidth": 0.74,
            "barWidth": 0.85, "xTickLabelRotation": -45 if len(table["data"]) > 20 else 0,
            "legend": legend, "tooltip": tooltip,
        }
    elif kind == "table":
        if not cuts:
            high = max(value for row in table["data"] for value in row[1:] if value is not None)
            cuts = [high * fraction for fraction in (0.2, 0.4, 0.6, 0.8)]
        thresholds = {
            "mode": "absolute",
            "steps": [{"value": value, "color": color} for value, color in zip([None, *cuts], colors)],
        }
        overrides.append(override(
            "byType", "number", color={"mode": "thresholds"}, thresholds=thresholds,
            **{"custom.width": 71, "custom.cellOptions": {"type": "color-background", "mode": "basic"}},
        ))
        overrides.append(override("byName", table["columns"][0], **{"custom.width": 160}))
        options = {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}
    else:
        raise ValueError(f"Unknown panel type: {kind}")
    return {"defaults": defaults, "overrides": overrides}, options


def raw_query(table):
    columns = table["columns"]
    types = table["types"]
    rows = table["data"]
    if not rows or len(columns) != len(types) or any(len(row) != len(columns) for row in rows):
        raise ValueError(f"Empty or malformed chart data: {columns}")
    fields = [{"name": name, "type": kind} for name, kind in zip(columns, types)]
    frame = {"schema": {"refId": "A", "fields": fields},
             "data": {"values": [list(values) for values in zip(*rows)]}}
    return {
        "refId": "A", "datasource": SOURCE, "scenarioId": "raw_frame",
        "rawFrameContent": json.dumps([frame], ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    }


def build_dashboard(snapshot):
    if snapshot["schema"] != 1:
        raise ValueError("Unsupported chart snapshot version")

    panels = []
    y = 0

    def section(title):
        nonlocal y
        panels.append({"id": len(panels) + 1, "type": "row", "title": title, "collapsed": False,
                       "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []})
        y += 1

    def chart(key, title, kind, x, width, height, *, description="", decimals=1,
              unit="none", horizontal=False, colors=None, cuts=()):
        table = snapshot["tables"][key]
        fields, options = panel_options(kind, table, decimals, unit, horizontal, colors, cuts)
        panels.append({
            "id": len(panels) + 1, "title": title, "type": kind, "description": description,
            "gridPos": {"x": x, "y": y, "w": width, "h": height},
            "datasource": SOURCE, "targets": [raw_query(table)], "fieldConfig": fields,
            "options": options,
        })

    section("Izzy · Возраст, лексика и темы")
    chart("age_emotions", "Эмоциональная лексика по возрасту", "table", 0, 24, 11, colors=WARM,
          description="Средняя частота слов категории NRC на 1000 токенов записи. Столбцы — возрастные группы Izzy.")
    y += 11
    chart("age_vader", "Тональность · VADER", "xychart", 0, 8, 9, decimals=3,
          description="Оценка compound от −1 до 1. По X — начало возрастной группы, например 18 означает 18–21.")
    chart("age_length", "Длина записи", "xychart", 8, 8, 9, decimals=0,
          description="Число токенов. По X — начало возрастной группы.")
    chart("age_mattr", "Словарь · MATTR-50", "xychart", 16, 8, 9, unit="percentunit",
          description="Средняя доля разных слов в скользящем окне из 50 токенов. По X — начало возрастной группы.")
    y += 9
    chart("age_topics", "Вес тем по возрасту", "table", 0, 24, 11, colors=COOL, unit="percentunit",
          description="Средний нормированный вес компонента NMF в каждой возрастной группе.")
    y += 11

    section("College · Длина, эмоции и слова")
    chart("gender_length", "Средняя длина записи", "barchart", 0, 8, 9, unit="suffix: токенов")
    chart("gender_emotions", "Эмоциональная лексика", "barchart", 8, 16, 9, unit=FREQUENCY,
          description="Средняя частота слов категории NRC на 1000 токенов по сериям college-f и college-m.")
    y += 9
    chart("words_common", "Слова, частые в обеих группах", "barchart", 0, 24, 14, horizontal=True,
          description="Число употреблений на 10 000 содержательных токенов. Выбраны 15 слов с наибольшей минимальной частотой в двух группах.")
    y += 14
    for key, title, x in (("words_female", "Относительно чаще у женщин", 0),
                          ("words_male", "Относительно чаще у мужчин", 12)):
        chart(key, title, "barchart", x, 12, 12, horizontal=True,
              description="Положительная разность нормированных частот на 10 000 содержательных токенов.")
    y += 12

    section("Izzy · Длина записи и лексика гнева")
    observed = snapshot["observed_slope"]
    chart("anger_null", f"Нулевые симуляции · наблюдаемый наклон {observed:.3f}", "barchart", 0, 24, 10, decimals=0,
          description=f"25 интервалов для 200 симуляций постоянной частоты. Наблюдаемый наклон логита по log(токенов): {observed:.3f}.")
    y += 10
    chart("anger_length", "Лексика гнева при одинаковой длине записи", "xychart", 0, 24, 10,
          unit=FREQUENCY, description="Частота слов NRC anger. Показаны группы от 30 снов. По X — начало возрастной группы.")
    y += 10

    section("Miami и Barb Sanders · Сравнение серий")
    chart("miami", "Miami · Сны с маркерами агрессии", "barchart", 0, 10, 10, unit="percent",
          description="Строгий словарь. Оценка доли и две границы 95% доверительного интервала Уилсона.")
    chart("barb", "Barb Sanders · Разность b2 − b", "barchart", 10, 14, 10, unit=FREQUENCY,
          description="Разность частот NRC и границы 95% доверительного интервала, полученного блочным бутстрэпом.")
    y += 10

    section("Norms · Совместная встречаемость категорий NRC")
    for key in ("norms-f", "norms-m"):
        chart(key, f"Отношение шансов · {key}", "table", 0, 24, 11, decimals=2, colors=OR_COLORS,
              cuts=(0.5, 0.8, 1.25, 2.0),
              description="Отношение шансов Мантеля–Хэнзеля с поправкой на длину. Нейтральное значение — 1. Шкала одинакова для обеих матриц.")
        y += 11

    section("Izzy · Тематическая модель с именами и без")
    chart("school", "Вес темы «Школа» по возрасту", "xychart", 0, 24, 10, unit="percentunit",
          description="Средний нормированный вес темы в двух моделях. По X — начало возрастной группы.")
    y += 10

    section("Статистическая мощность · Минимальная обнаружимая разность")
    for effect, x in (("1", 0), ("1.5", 8), ("2", 16)):
        chart(f"mde_{effect}", f"Дизайн-эффект {effect.replace('.', ',')}", "xychart", x, 8, 10,
              unit="suffix: п.п.", description="По X — базовая доля в процентах. По Y — минимальная обнаружимая разность при мощности 80%, α = 0,05.")

    return {
        "uid": DASH_UID, "title": "DreamPoly · Анализ сновидений",
        "description": "Агрегаты из notebooks/DreamPoly.ipynb на данных DreamBank. Возрастные группы и серии корпуса.",
        "schemaVersion": 41, "editable": True, "tags": ["DreamBank", "DreamPoly"],
        "timepicker": {"hidden": True}, "panels": panels,
        "links": [{"title": "Архив DreamBank на Zenodo", "type": "link",
                   "url": "https://zenodo.org/records/18159468", "targetBlank": True}],
    }


def publish(path: Path):
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    dashboard = build_dashboard(snapshot)
    token = os.environ.get("GRAFANA_TOKEN")
    if token:
        authorization = f"Bearer {token}"
    else:
        username = os.environ.get("GRAFANA_USER", "admin")
        password = os.environ.get("GRAFANA_PASSWORD")
        if password is None:
            password = getpass(f"Пароль Grafana ({username}): ")
        encoded = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        authorization = f"Basic {encoded}"

    opener = build_opener(ProxyHandler({}))

    def request(route, method="GET", payload=None, missing_ok=False):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if payload else None
        req = Request(ROOT + route, data=body, method=method, headers={"Content-Type": "application/json"})
        req.add_unredirected_header("Authorization", authorization)
        try:
            with opener.open(req, timeout=25) as response:
                return json.load(response)
        except HTTPError as error:
            if missing_ok and error.code == 404:
                return None
            details = error.read().decode("utf-8", errors="replace")
            raise SystemExit(f"Grafana HTTP {error.code}: {details}") from error
        except URLError as error:
            raise SystemExit(f"Grafana недоступна по адресу {ROOT}: {error.reason}") from error

    source_route = f"/api/datasources/uid/{SOURCE['uid']}"
    existing_source = request(source_route, missing_ok=True)
    if existing_source is None:
        request("/api/datasources", "POST", {**SOURCE, "name": "DreamPoly · Данные ноутбука",
                                             "isDefault": False, "access": "proxy"})
    elif existing_source["type"] != SOURCE["type"]:
        raise SystemExit(f"Другой источник уже использует идентификатор {SOURCE['uid']}.")

    collection = "/apis/dashboard.grafana.app/v1/namespaces/default/dashboards"
    route = f"{collection}/{DASH_UID}"
    resource = {"apiVersion": "dashboard.grafana.app/v1", "kind": "Dashboard",
                "metadata": {"name": DASH_UID}, "spec": dashboard}
    previous = request(route, missing_ok=True)
    if previous is None:
        request(collection, "POST", resource)
    else:
        resource["metadata"] = previous["metadata"]
        request(route, "PUT", resource)
    saved = request(route)
    expected = [(panel["id"], panel.get("targets")) for panel in dashboard["panels"]]
    actual = [(panel["id"], panel.get("targets")) for panel in saved["spec"]["panels"]]
    if actual != expected:
        raise SystemExit("Grafana вернула неполный дашборд или изменила данные панелей.")
    return ROOT + request(f"/api/dashboards/uid/{DASH_UID}")["meta"]["url"]
