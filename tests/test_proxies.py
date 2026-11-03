import random
import time

import aiohttp
import pytest

from baidu_reverse import BaiduBlockedError, BaiduChat, ProxyManager, ProxyPool
from baidu_reverse.client import _proxy_var
from baidu_reverse.proxies import _extract_proxies

from conftest import ScriptedChat


# --------------------------------------------------------------------------- scraping helpers
def test_extract_proxies_validates_ip_and_port():
    text = "1.2.3.4:8080\n300.1.1.1:80\n5.6.7.8:99999\nfoo 9.9.9.9:3128 bar\n1.2.3.4:8080\n10.0.0.1:0"
    assert _extract_proxies(text) == ["1.2.3.4:8080", "9.9.9.9:3128"]


# --------------------------------------------------------------------------- ProxyPool
def test_empty_pool_means_direct():
    assert ProxyPool().next() is None and ProxyPool().stats()["healthy"] == 0


def test_network_failure_bans_then_success_forgives():
    pool = ProxyPool(["a:1", "b:2"], fail_cooldown=60)
    pool.report_failure("a:1")
    assert {pool.next() for _ in range(30)} == {"b:2"}          # a is resting
    pool.report_success("a:1", latency=1.0)
    assert "a:1" in {pool.next() for _ in range(200)}            # back in rotation


def test_proxy_that_never_worked_is_dropped_after_max_fails():
    pool = ProxyPool(["a:1"], max_fails=3)
    for _ in range(3):
        pool.report_failure("a:1")
    assert "a:1" not in pool
    # ...but one that has worked before only rests
    pool = ProxyPool(["w:1"], max_fails=3)
    pool.report_success("w:1")
    for _ in range(5):
        pool.report_failure("w:1")
    assert "w:1" in pool and pool.next() is None


def test_blocked_exit_rests_much_longer_than_a_failure():
    pool = ProxyPool(["a:1"], fail_cooldown=30, block_cooldown=600)
    pool.report_success("a:1")
    pool.report_failure("a:1")
    short = pool._proxies["a:1"].banned_until - time.time()
    pool.report_success("a:1")
    pool.report_blocked("a:1")
    long = pool._proxies["a:1"].banned_until - time.time()
    assert long > 10 * short and pool._proxies["a:1"].blocked == 1


def test_rotation_prefers_fast_proxies_but_uses_all():
    pool = ProxyPool(rng=random.Random(7))
    pool.add("fast:1", 0.5)
    pool.add("slow:1", 5.0)
    picks = [pool.next() for _ in range(400)]
    for p in pool._proxies.values():
        p.last_used = 0.0
    assert picks.count("fast:1") > picks.count("slow:1") > 0


def test_consecutive_picks_spread_over_the_pool():
    pool = ProxyPool([f"p{i}:1" for i in range(6)], rng=random.Random(1))
    first = [pool.next() for _ in range(6)]
    assert len(set(first)) >= 4                                   # "recently used" is penalised


# --------------------------------------------------------------------------- ProxyManager (no network)
def make_manager(tmp_path, alive, scraped, **kw):
    mgr = ProxyManager(cache_file=str(tmp_path / "p.json"), check_concurrency=20, **kw)
    calls = {"checked": [], "scrapes": 0}

    async def fake_scrape():
        calls["scrapes"] += 1
        return list(scraped)

    async def fake_check(addr, session=None):
        calls["checked"].append(addr)
        return 1.0 + (hash(addr) % 10) / 10 if addr in alive else None

    mgr.scrape, mgr.check = fake_scrape, fake_check
    return mgr, calls


@pytest.mark.asyncio
async def test_refill_stops_as_soon_as_enough_are_healthy(tmp_path):
    scraped = [f"10.0.{i // 250}.{i % 250}:80" for i in range(1000)]
    alive = set(scraped[::2])                                      # 50% alive
    mgr, calls = make_manager(tmp_path, alive, scraped, target_size=10, max_candidates=1000)
    summary = await mgr.refill()
    assert 10 <= len(mgr.healthy()) <= 10 + 20 + 1                 # overshoot bounded by in-flight checks
    assert len(calls["checked"]) < 700                             # did NOT check all 1000
    assert summary["healthy"] == len(mgr.healthy())


@pytest.mark.asyncio
async def test_refill_remembers_dead_candidates_and_skips_them(tmp_path):
    scraped = [f"9.9.9.{i}:80" for i in range(30)]
    mgr, calls = make_manager(tmp_path, set(), scraped, target_size=5)
    await mgr.refill()
    first_round = len(calls["checked"])
    assert first_round == 30 and not mgr.healthy()
    await mgr.refill()
    assert len(calls["checked"]) == first_round                    # nothing re-checked: all known dead


@pytest.mark.asyncio
async def test_cache_survives_restart_and_is_reverified(tmp_path):
    scraped = ["1.1.1.1:80", "2.2.2.2:80", "3.3.3.3:80"]
    mgr, _ = make_manager(tmp_path, set(scraped), scraped, target_size=3)
    await mgr.refill()
    assert (tmp_path / "p.json").exists()

    # new process: scraping returns nothing, one cached proxy died meanwhile
    mgr2, calls2 = make_manager(tmp_path, {"1.1.1.1:80", "3.3.3.3:80"}, [], target_size=3)
    await mgr2.refill()
    assert sorted(p.addr for p in mgr2.healthy()) == ["1.1.1.1:80", "3.3.3.3:80"]
    assert set(calls2["checked"]) == set(scraped)                  # cached ones were re-verified, not trusted


@pytest.mark.asyncio
async def test_stale_cache_is_ignored(tmp_path):
    mgr, _ = make_manager(tmp_path, {"1.1.1.1:80"}, ["1.1.1.1:80"], target_size=1)
    await mgr.refill()
    mgr2, calls2 = make_manager(tmp_path, set(), [], target_size=1, cache_ttl=0.01)
    time.sleep(0.05)
    await mgr2.refill()
    assert calls2["checked"] == []


@pytest.mark.asyncio
async def test_ensure_refills_inline_when_empty_and_in_background_when_low(tmp_path):
    scraped = [f"7.7.7.{i}:80" for i in range(20)]
    mgr, calls = make_manager(tmp_path, set(scraped), scraped, target_size=8, min_healthy=4)
    await mgr.ensure()                                             # empty -> waits for the refill
    assert len(mgr.healthy()) >= 8
    for p in list(mgr.healthy())[:6]:                              # burn most of them
        mgr.report_blocked(p.addr)
    assert len(mgr.healthy()) < mgr.min_healthy
    n = calls["scrapes"]
    await mgr.ensure()                                             # low -> background task, returns at once
    assert mgr._bg is not None
    await mgr._bg
    assert calls["scrapes"] == n + 1 and len(mgr.healthy()) >= 8


# --------------------------------------------------------------------------- client integration
class FakeResp:
    def __init__(self, status=200):
        self.status = status
        self.released = False

    def release(self):
        self.released = True


class FakeSession:
    """Raises/answers per proxy, like a pool of flaky exits."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.used = []

    async def request(self, method, url, proxy=None, **kw):
        self.used.append(proxy)
        what = self.behaviour.get(proxy)
        if what == "dead":
            raise aiohttp.ClientProxyConnectionError(None, OSError("refused"))
        if what == "timeout":
            raise TimeoutError()
        return FakeResp(what if isinstance(what, int) else 200)


@pytest.mark.asyncio
async def test_send_fails_over_to_a_working_proxy_and_reports_bad_ones():
    pool = ProxyPool(["1.1.1.1:80", "2.2.2.2:80", "3.3.3.3:80"], rng=random.Random(3))
    chat = BaiduChat(proxy_pool=pool)
    sess = FakeSession({"http://1.1.1.1:80": "dead", "http://2.2.2.2:80": 502, "http://3.3.3.3:80": 200})
    resp = await chat._send(sess, "GET", "https://x/")
    assert resp.status == 200 and sess.used[-1] == "http://3.3.3.3:80"
    assert pool._proxies["1.1.1.1:80"].fail >= 1 or "1.1.1.1:80" not in pool   # reported, then rested/dropped
    assert sess.used.count("http://3.3.3.3:80") == 1


@pytest.mark.asyncio
async def test_send_gives_up_after_proxy_retries():
    pool = ProxyPool([f"{i}.{i}.{i}.{i}:80" for i in range(1, 9)])
    chat = BaiduChat(proxy_pool=pool, proxy_retries=3)
    sess = FakeSession({f"http://{i}.{i}.{i}.{i}:80": "timeout" for i in range(1, 9)})
    from baidu_reverse import BaiduError

    with pytest.raises(BaiduError, match="all 4 proxy attempts failed"):
        await chat._send(sess, "GET", "https://x/")
    assert len(sess.used) == 4


@pytest.mark.asyncio
async def test_without_a_pool_errors_propagate_untouched():
    chat = BaiduChat(proxy="http://9.9.9.9:80")
    sess = FakeSession({"http://9.9.9.9:80": "dead"})
    with pytest.raises(aiohttp.ClientProxyConnectionError):
        await chat._send(sess, "GET", "https://x/")
    assert sess.used == ["http://9.9.9.9:80"]


@pytest.mark.asyncio
async def test_block_rotates_the_exit_before_touching_the_captcha():
    pool = ProxyPool(["1.1.1.1:80", "2.2.2.2:80"])
    solved = []
    chat = ScriptedChat(lambda q, h, s: "fine", proxy_pool=pool, proxy_retries=2, auto_solve=True, max_block_retries=1)

    async def fake_solve():
        solved.append(1)
        return ""

    chat.solve_captcha = fake_solve
    _proxy_var.set("1.1.1.1:80")
    assert await chat._recover_from_block(0, "kunlun") is True       # rotation #1
    assert pool._proxies["1.1.1.1:80"].blocked == 1                    # that exit is now resting long
    assert pool._proxies["1.1.1.1:80"].banned_until - time.time() > 300
    assert await chat._recover_from_block(1, "kunlun") is True       # rotation #2
    assert solved == []
    assert await chat._recover_from_block(2, "kunlun") is True       # rotations exhausted -> captcha
    assert solved == [1]
    assert await chat._recover_from_block(3, "kunlun") is False       # and then give up


@pytest.mark.asyncio
async def test_blocked_stream_retries_through_other_exits_then_raises():
    pool = ProxyPool(["1.1.1.1:80"])
    chat = ScriptedChat(lambda q, h, s: "fine", proxy_pool=pool, proxy_retries=2)
    chat.block_next = 10
    with pytest.raises(BaiduBlockedError):
        await chat.generate("hi")
    assert len(chat.sent) == 3                                         # first try + 2 rotations, no auto_solve


@pytest.mark.asyncio
async def test_rotation_recovers_when_a_later_exit_is_not_blocked():
    chat = ScriptedChat(lambda q, h, s: "fine", proxy_pool=ProxyPool(["1.1.1.1:80"]), proxy_retries=3)
    chat.block_next = 2
    assert await chat.generate("hi") == "fine"
    assert len(chat.sent) == 3


def test_a_static_pool_and_rotate_flag_are_wired():
    assert isinstance(BaiduChat(proxies=["1.1.1.1:80"]).proxy_pool, ProxyPool)
    assert isinstance(BaiduChat(rotate_proxies=True).proxy_pool, ProxyManager)
    assert BaiduChat(proxy="http://x:1").proxy_pool is None
    mine = ProxyPool(["1.1.1.1:80"])
    assert BaiduChat(proxy_pool=mine).proxy_pool is mine and BaiduChat(proxy_pool=mine)._owns_pool is False
