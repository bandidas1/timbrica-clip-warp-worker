"""Самопроверка расписания сцен — без GPU, без torch.

Зачем: по этим числам приложение вклеивает пересъёмку сцены на место
(App\\Services\\AiVideo\\ClipPlan::shotFrames — порт `_schedule`). Ошибка на
кадр — и на стыке мигает кадр соседней сцены. Поэтому:

  python schedule_selftest.py            — проверить `_render_range` на случаях;
  python schedule_selftest.py --emit F   — записать эталоны в JSON F; их читает
                                           PHP-сторож tests/Feature/ClipRedoTest.php.

Функции берутся из handler.py разбором исходника (ast), а не импортом: импорт
тянет torch и diffusers, которых в CI нет.
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(names):
    src = open(os.path.join(HERE, "handler.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {}
    # Функции видят друг друга через общий словарь (_schedule зовёт _plan_walk).
    exec(compile(ast.Module(body=keep, type_ignores=[]), "handler.py", "exec"), ns)
    return [ns[n] for n in names]


_schedule, _render_range, _plan_walk, _shot_seeds, _render_ranges = _load(
    ["_schedule", "_render_range", "_plan_walk", "_shot_seeds", "_render_ranges"])

# Случаи — то, что приходит на самом деле, и края: план короче и длиннее клипа,
# пустые и битые длины, половина кадра (банковское округление Python), не-словарь.
CASES = [
    {"seconds": 60, "fps": 12, "shot_seconds": 8, "shots": None},
    {"seconds": 61, "fps": 12, "shot_seconds": 7.5, "shots": None},
    {"seconds": 8, "fps": 12, "shot_seconds": 15, "shots": None},
    {"seconds": 30, "fps": 12, "shot_seconds": 8, "shots": [
        {"prompt": "a", "seconds": 7.63}, {"prompt": "", "seconds": 7.62},
        {"prompt": "b", "seconds": 7.63}, {"prompt": "c", "seconds": 7.12}]},
    {"seconds": 30, "fps": 12, "shot_seconds": 8, "shots": [{"seconds": 10}, {"seconds": 5}]},
    {"seconds": 20, "fps": 12, "shot_seconds": 6, "shots": [{"seconds": 9}, {"seconds": 9}, {"seconds": 9}, {"seconds": 9}]},
    {"seconds": 20, "fps": 12, "shot_seconds": 6, "shots": [{"seconds": 0}, {"seconds": -3}, {"seconds": "x"}, {"prompt": "only"}]},
    {"seconds": 20, "fps": 10, "shot_seconds": 6, "shots": [{"seconds": 4.25}, {"seconds": 4.35}, {"seconds": 4.45}]},
    {"seconds": 20, "fps": 12, "shot_seconds": 6, "shots": ["junk", {"seconds": 5}, 7, {"seconds": 5}]},
    {"seconds": 180, "fps": 12, "shot_seconds": 8, "shots": [{"seconds": round(4 + (k % 5) * 1.37, 2)} for k in range(60)]},
    {"seconds": 3, "fps": 12, "shot_seconds": 4, "shots": [{"seconds": 0.1}]},
]


def ranges(case):
    total = max(int(round(float(case["seconds"]) * case["fps"])), 2)
    shot_len = max(int(round(float(case["shot_seconds"]) * case["fps"])), 2)
    plan = [s for s in (case["shots"] or []) if isinstance(s, dict)]
    starts, _ = _schedule(total, case["fps"], shot_len, plan, "base")
    out = []
    k = 1
    while True:
        lo, hi = _render_range(starts, total, k)
        if lo is None:
            break
        out.append([lo, hi])
        k += 1
    return total, out


def main():
    rows = []
    for c in CASES:
        total, rs = ranges(c)
        # Сцены покрывают клип без дыр и наложений, начинаются с 0.
        assert rs and rs[0][0] == 0 and rs[-1][1] == total, (c, rs)
        for (a0, b0), (a1, b1) in zip(rs, rs[1:]):
            assert b0 == a1 and a0 < b0, (c, rs)
        assert _render_range(set(r[0] for r in rs), total, 0) == (0, total)
        assert _render_range(set(r[0] for r in rs), total, len(rs) + 1) == (None, None)
        starts = set(r[0] for r in rs)
        # Весь клип и одна сцена — как до пачки (старые задания ведут себя так же).
        assert _render_ranges(starts, total, 0, None) == [(0, total, 0, None)]
        assert _render_ranges(starts, total, 1, 77) == [(rs[0][0], rs[0][1], 1, 77)]
        assert _render_ranges(starts, total, len(rs) + 1, None) is None
        # Несколько сцен: по порядку клипа, без повторов, сид — свой у каждой,
        # `render_shots` перекрывает `render_shot`; чужой номер — отказ всей пачке.
        if len(rs) >= 3:
            got = _render_ranges(starts, total, 1, 5, [3, 1, 3], {"1": 11, "3": 33})
            assert got == [(rs[0][0], rs[0][1], 1, 11), (rs[2][0], rs[2][1], 3, 33)], (c, got)
            got = _render_ranges(starts, total, 3, None, [len(rs), 2], {})
            assert got == [(rs[1][0], rs[1][1], 2, None), (rs[-1][0], rs[-1][1], len(rs), None)], (c, got)
            assert _render_ranges(starts, total, 1, None, [2, len(rs) + 1], {}) is None
            assert _render_ranges(starts, total, 1, None, [True], {}) is None
        rows.append({"params": c, "frames": rs})
        # Свой сид шота (раскадровка) ложится РОВНО на кадр начала того же шота,
        # что и в расписании: иначе клип начал бы сцену не с показанного кадра.
        if c["shots"]:
            fps, shot_len = c["fps"], max(int(round(c["shot_seconds"] * c["fps"])), 2)
            plan = [dict(s, seed=1000 + k) if isinstance(s, dict) else s for k, s in enumerate(c["shots"])]
            seeds = _shot_seeds(total, fps, shot_len, [s for s in plan if isinstance(s, dict)])
            walked = [start for start, _l, _s in _plan_walk(total, fps, shot_len, [s for s in plan if isinstance(s, dict)])]
            assert sorted(seeds) == walked and set(walked) <= set(r[0] for r in rs), (c, seeds, rs)
    if "--emit" in sys.argv:
        path = sys.argv[sys.argv.index("--emit") + 1]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
            f.write("\n")
    print("schedule selftest: ok, %d cases" % len(rows))


if __name__ == "__main__":
    main()
