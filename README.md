### Секция 11. Расширенный статистический анализ

Считается в общем прогоне `python -m dreambank`:

- `Aggression`    — лексические маркеры агрессии (широкий/строгий словарь),
                    сравнение norms/college/miami, MDE и Fisher-power
- `LengthAnger`   — длина сна и частота лексики гнева (пуассоновская
                    модель с offset)
- `IzzyStability` — устойчивость показателей внутри возрастных папок
- `Cooccurrence`  — совместная встречаемость категорий NRC
- `BarbSanders`   — половины серии `b`, `b` vs `b2`
- `NamesTopics`   — влияние имён на тематическую модель Izzy

Результаты: `results/DD_MM_YYYY_HH_MM/extended/` (csv)
и `figures/extended/` (png). Описание методов — в `dreampoly_extended.py`
и `extended_pipeline.py`.