"""Самопроверка доставки готового клипа — без GPU, без torch, без сети.

Зачем: 28.10.2026 три клипа 1080p по 226 с упали `upload_failed:
ConnectionError` после десяти минут отрисовки каждый — файл уходил одним POST,
а сеть до Бегета в тот час зависала. Теперь `_deliver` шлёт файл порциями
протоколом смещений (ClipUploadController::storeSlice) и повторяет порцию, а
не отрисовку. Здесь — поддельная дверь с той же семантикой, что
App\\Support\\OffsetAppend, и сценарии того вечера:

  python upload_selftest.py

Функции берутся из handler.py разбором исходника (ast), а не импортом: импорт
тянет torch и diffusers, которых в CI нет.
"""
import ast
import hashlib
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["_deliver", "_upload_retryable", "_upload_pause", "_upload_send",
         "_deliver_slices", "_deliver_whole"]


def _load():
    src = open(os.path.join(HERE, "handler.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "handler.py", "exec"), ns)
    missing = [n for n in NAMES if n not in ns]
    assert not missing, f"в handler.py нет {missing}"
    return ns


class ConnectionError(Exception):   # noqa: A001 — имя как у requests, его и пишет расписка
    pass


class Resp:
    def __init__(self, code, body):
        self.status_code = code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


class Door:
    """Дверь приёма: части по смещению, `received` в каждом ответе.

    `faults` — что случится с N-м запросом (с нуля):
      'drop'   — обрыв до сервера (ничего не легло), как 408 после тишины;
      '408'    — nginx ответил 408, ничего не легло;
      'lost'   — сервер ПРИНЯЛ порцию, а клиент ответа не увидел;
      '403'    — отказ подписи.
    """

    def __init__(self, faults=None, clock=None):
        self.faults = dict(faults or {})
        self.part = b""
        self.final = None
        self.calls = 0
        self.clock = clock

    def post(self, url, headers=None, timeout=None, data=None, files=None):
        n = self.calls
        self.calls += 1
        if self.clock is not None:
            self.clock.t += 1.0                     # каждая порция — секунда на ссылке
        fault = self.faults.get(n)
        if fault == "drop":
            raise ConnectionError("Connection aborted.")
        if fault == "408":
            return Resp(408, {})
        if fault == "403":
            return Resp(403, {"ok": False, "error": "forbidden"})
        if files and "file" in files:               # цельная загрузка (образ до 29.10)
            self.final = files["file"][1]
            return Resp(200, {"ok": True, "done": True})
        offset, size = int(data["offset"]), int(data["size"])
        chunk = files["chunk"][1]
        if self.final is not None and len(self.final) == size:
            resp = Resp(200, {"ok": True, "done": True})
        elif offset > len(self.part):
            resp = Resp(409, {"ok": False, "error": "gap", "received": len(self.part)})
        else:
            skip = len(self.part) - offset
            if skip < len(chunk):
                self.part += chunk[skip:]
            if len(self.part) == size:
                assert hashlib.sha256(self.part).hexdigest() == data["sha256"], "собран не тот файл"
                self.final, self.part = self.part, b""
                resp = Resp(200, {"ok": True, "done": True})
            else:
                resp = Resp(200, {"ok": True, "received": len(self.part)})
        if fault == "lost":
            raise ConnectionError("Read timed out.")
        return resp


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(ns, door, clock, mp4, chunk=1000):
    up = {"url": "https://x/internal/clip/upload", "gen_id": 7, "expires": 1, "sig": "s"}
    if chunk:
        up["chunk_bytes"] = chunk
    return ns["_deliver"](mp4, up, post=door.post, sleep=clock.sleep, clock=clock)


def main():
    ns = _load()
    rnd = random.Random(1028)
    mp4 = bytes(rnd.getrandbits(8) for _ in range(10_500))
    sha = hashlib.sha256(mp4).hexdigest()
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(("ok   " if ok else "FAIL ") + name + (f" — {detail}" if detail and not ok else ""))
        fails += 0 if ok else 1

    # 1. Здоровая сеть: 11 порций, файл целый.
    c = Clock()
    d = Door(clock=c)
    r = run(ns, d, c, mp4)
    check("порции собирают тот же файл", r.get("stored") and d.final == mp4 and r["sha256"] == sha, r)
    check("здоровая сеть — без повторов", r.get("upload_retries") == 0 and d.calls == 11, (r, d.calls))

    # 2. Вечер 28.10: обрывы, 408 и «принял, а ответа не видно» вперемешку.
    c = Clock()
    d = Door(faults={1: "drop", 2: "408", 4: "lost", 5: "drop", 9: "408", 12: "lost"}, clock=c)
    r = run(ns, d, c, mp4)
    check("обрывы и 408 стоят порции, а не отрисовки", r.get("stored") and d.final == mp4, r)
    check("повторы посчитаны в расписке", r.get("upload_retries", 0) >= 4, r)

    # 3. Принятая, но «потерянная» порция не ложится дважды (сервер пропускает
    #    уже имеющиеся байты, клиент продолжает с `received`).
    c = Clock()
    d = Door(faults={0: "lost", 3: "lost"}, clock=c)
    r = run(ns, d, c, mp4)
    check("повтор принятой порции не удваивает байты", d.final == mp4, (len(d.final or b""), len(mp4)))

    # 4. Сервер помнит меньше, чем клиент думает (gap) — продолжить с его числа.
    c = Clock()
    d = Door(clock=c)
    d.part = mp4[:2500]
    d.faults = {}
    up = {"url": "u", "gen_id": 1, "expires": 1, "sig": "s", "chunk_bytes": 1000}
    # Начинаем «с середины», как воркер после перезапуска опроса.
    r = ns["_deliver_slices"]("u", {}, mp4, sha, 1000, post=d.post, sleep=c.sleep, clock=c,
                              stall_s=360, total_s=900)
    check("gap: клиент берёт `received` сервера", r.get("stored") and d.final == mp4, r)

    # 5. Подпись отвергнута — повтор ничего не изменит, ответ сразу.
    c = Clock()
    d = Door(faults={0: "403"}, clock=c)
    r = run(ns, d, c, mp4)
    check("403 — сразу, без повторов", str(r.get("error", "")).startswith("upload_http_403") and d.calls == 1, (r, d.calls))

    # 6. Сеть лежит совсем — сдаёмся по терпению, а не висим до потолка RunPod.
    c = Clock()
    d = Door(faults={i: "drop" for i in range(10_000)}, clock=c)
    r = run(ns, d, c, mp4)
    check("мёртвая сеть — отказ по терпению", r.get("error") == "upload_failed: ConnectionError", r)
    check("терпение ~6 мин без продвижения, не дольше 15", 360 <= c.t <= 900, c.t)

    # 7. Медленный, но живой час: продвижение сбрасывает терпение.
    c = Clock()
    faults = {i: "drop" for i in range(0, 200) if i % 3 != 2}   # две из трёх порций рвутся
    d = Door(faults=faults, clock=c)
    r = run(ns, d, c, mp4)
    check("две из трёх порций рвутся — файл всё равно доезжает", r.get("stored") and d.final == mp4, (r, c.t))

    # 8. Приложение до 29.10 (нет chunk_bytes): цельный файл, два повтора.
    c = Clock()
    d = Door(faults={0: "408", 1: "drop"}, clock=c)
    r = run(ns, d, c, mp4, chunk=0)
    check("цельная загрузка повторяется", r.get("stored") and d.final == mp4 and d.calls == 3, (r, d.calls))
    c = Clock()
    d = Door(faults={0: "drop", 1: "drop", 2: "drop", 3: "drop"}, clock=c)
    r = run(ns, d, c, mp4, chunk=0)
    check("цельная загрузка сдаётся после трёх попыток", r.get("error") == "upload_failed: ConnectionError" and d.calls == 3, (r, d.calls))

    print("upload_selftest: " + ("FAIL" if fails else "all good"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
