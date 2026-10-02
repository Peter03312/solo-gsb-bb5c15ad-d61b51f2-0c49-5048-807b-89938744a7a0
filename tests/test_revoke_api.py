"""HTTP API 集成测试：证书撤销（revoke）与窗口当前可用证书。

覆盖：
- 撤销：只改变有效性（版本/结论/快照/issued_at 原样），响应与查询带
  撤销时间与原因；撤销历史证书（非最新版）允许；
- 幂等与冲突：同原因重复撤销返回原记录（不产生新事件）；原因不同 409；
  空白原因 422；缺原因 422；证书不存在 404；
- 当前可用证书：只有最新签发版本未撤销时才可用；从未签发 no_certificate；
  撤销最新版 latest_revoked 且不自动退回旧版；再次签发沿用递增版本并成为
  当前可用证书；
- 历史可读：证书查询、窗口证书列表、窗口详情、修订影响报告均可读取历史
  证书并标示撤销状态（报告仍只比对不可变快照）；
- 并发：撤销与签发并发，结果符合实际提交顺序（两种串行化次序均自洽）。

运行：PYTHONPATH=. python -m pytest tests/test_revoke_api.py -q
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.config import settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(settings, "db_path", str(db_file))
    monkeypatch.setattr(settings, "db_timeout", 5)
    # 延迟导入 app，确保 connect() 读取被覆盖后的 settings
    from app.main import app

    with TestClient(app) as c:
        yield c


WINDOW_BASE = "2026-10-01T00:00:00Z"


def make_window(client, start=WINDOW_BASE, minutes=60, tmin=18.0, tmax=22.0,
                max_exceed=300.0, max_gap=600.0, name=None):
    body = {
        "start_at": start,
        "end_at": (datetime.fromisoformat(start.replace("Z", "+00:00"))
                   + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z"),
        "allowed_temp_min": tmin,
        "allowed_temp_max": tmax,
        "max_continuous_exceed_seconds": max_exceed,
        "max_sample_interval_seconds": max_gap,
    }
    if name:
        body["name"] = name
    r = client.post("/api/v1/windows", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def reading(rid, rev, minute, temp):
    ts = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return {
        "reading_id": rid, "revision": rev,
        "measured_at": ts.isoformat().replace("+00:00", "Z"),
        "temperature": temp,
    }


def issue(client, wid=1, note=None):
    return client.post(f"/api/v1/windows/{wid}/certificates/issue", json={"note": note})


def revoke(client, cert_id, reason):
    return client.post(f"/api/v1/certificates/{cert_id}/revoke",
                       json={"reason": reason})


def seed_issued(client, versions=1):
    """建窗、一条读数，签发 versions 张证书（每次改一点温度产生新版本）。"""
    make_window(client)
    client.post("/api/v1/windows/1/readings",
                json=reading("a", 1, 0, 20.0))
    certs = [issue(client, note=f"v{i}").json() for i in range(1, versions + 1)]
    return certs


# ---------------------------------------------------------------- 撤销基本语义

def test_revoke_only_changes_validity_keeps_version_verdict_snapshot(client):
    c1 = seed_issued(client)[0]
    before = client.get(f"/api/v1/certificates/{c1['id']}").json()

    r = revoke(client, c1["id"], "依据有误：读数来源错误")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["effect"] == "revoked"

    rev = body["revocation"]
    assert rev["certificate_id"] == c1["id"]
    assert rev["window_id"] == 1
    assert rev["version"] == 1
    assert rev["reason"] == "依据有误：读数来源错误"
    assert rev["revoked_at"]  # 撤销时间已留存

    cert = body["certificate"]
    # 撤销只改变有效性：版本、结论、签发时间、备注、快照均原样保留
    assert cert["revoked"] is True and cert["status"] == "revoked"
    assert cert["version"] == before["version"]
    assert cert["verdict"] == before["verdict"]
    assert cert["verdict_label_zh"] == before["verdict_label_zh"]
    assert cert["issued_at"] == before["issued_at"]
    assert cert["note"] == before["note"]
    assert cert["snapshot"] == before["snapshot"]
    assert cert["revocation"] == rev

    # 按证书主键查询：历史证书仍可读，且标示撤销状态与时间/原因
    got = client.get(f"/api/v1/certificates/{c1['id']}").json()
    assert got["revoked"] is True and got["status"] == "revoked"
    assert got["version"] == 1 and got["snapshot"] == before["snapshot"]
    assert got["revocation"] == rev


def test_revocation_query_endpoint(client):
    c1 = seed_issued(client)[0]
    # 未撤销：revoked=false、revocation=null
    r = client.get(f"/api/v1/certificates/{c1['id']}/revocation")
    assert r.status_code == 200
    body = r.json()
    assert body["revoked"] is False and body["status"] == "valid"
    assert body["revocation"] is None
    assert body["certificate"]["id"] == c1["id"]

    revoke(client, c1["id"], "依据有误")
    r = client.get(f"/api/v1/certificates/{c1['id']}/revocation")
    body = r.json()
    assert body["revoked"] is True and body["status"] == "revoked"
    assert body["revocation"]["reason"] == "依据有误"
    assert body["revocation"]["revoked_at"]
    assert body["version"] == 1

    # 撤销查询：证书不存在 -> 404
    assert client.get("/api/v1/certificates/999/revocation").status_code == 404


def test_revoke_missing_certificate_404(client):
    seed_issued(client)
    r = revoke(client, 42, "依据有误")
    assert r.status_code == 404 and "不存在" in r.json()["detail"]


def test_revoke_blank_reason_422(client):
    c1 = seed_issued(client)[0]
    for bad in ("", "   ", "\n\t "):
        r = revoke(client, c1["id"], bad)
        assert r.status_code == 422, bad
    # 缺 reason 字段同样 422
    r = client.post(f"/api/v1/certificates/{c1['id']}/revoke", json={})
    assert r.status_code == 422
    # 撤销未生效
    got = client.get(f"/api/v1/certificates/{c1['id']}").json()
    assert got["revoked"] is False


def test_revoke_older_version_allowed(client):
    """撤销依据有误可针对任意已签发证书；非最新版也可撤销。"""
    c1, c2 = seed_issued(client, versions=2)
    r = revoke(client, c1["id"], "旧版依据有误")
    assert r.status_code == 201 and r.json()["effect"] == "revoked"
    # 旧版撤销不影响最新版有效性：当前可用仍是 v2
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is True and cur["certificate"]["id"] == c2["id"]
    got1 = client.get(f"/api/v1/certificates/{c1['id']}").json()
    got2 = client.get(f"/api/v1/certificates/{c2['id']}").json()
    assert got1["revoked"] is True and got2["revoked"] is False


# ------------------------------------------------------------ 幂等与冲突

def test_revoke_same_reason_replay_idempotent_returns_original_record(client):
    c1 = seed_issued(client)[0]
    r1 = revoke(client, c1["id"], "依据有误：原始记录已撤回")
    assert r1.status_code == 201 and r1.json()["effect"] == "revoked"
    first_rev = r1.json()["revocation"]

    r2 = revoke(client, c1["id"], "依据有误：原始记录已撤回")
    assert r2.status_code == 201
    body2 = r2.json()
    # 同原因重复撤销返回原记录，不产生新事件（撤销时间不变）
    assert body2["effect"] == "idempotent"
    assert body2["revocation"] == first_rev
    assert body2["certificate"]["revoked"] is True

    # 撤销记录只有一条
    from app.database import connect
    conn = connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM certificate_revocations WHERE certificate_id=?",
            (c1["id"],),
        ).fetchone()["n"]
        assert n == 1
    finally:
        conn.close()

    # 再换第三、四次相同原因重放，仍是原记录
    r3 = revoke(client, c1["id"], "依据有误：原始记录已撤回")
    assert r3.json()["effect"] == "idempotent"
    assert r3.json()["revocation"] == first_rev


def test_revoke_different_reason_conflict_409(client):
    c1 = seed_issued(client)[0]
    revoke(client, c1["id"], "第一个原因")
    r = revoke(client, c1["id"], "第二个原因")
    assert r.status_code == 409 and "原因冲突" in r.json()["detail"]

    # 原撤销记录不变（原因与时间仍是第一次的）
    got = client.get(f"/api/v1/certificates/{c1['id']}/revocation").json()
    assert got["revocation"]["reason"] == "第一个原因"

    from app.database import connect
    conn = connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM certificate_revocations WHERE certificate_id=?",
            (c1["id"],),
        ).fetchone()["n"]
        assert n == 1
    finally:
        conn.close()

    # 冲突后用原原因重放仍然幂等
    r = revoke(client, c1["id"], "第一个原因")
    assert r.status_code == 201 and r.json()["effect"] == "idempotent"


# ------------------------------------------------------------ 当前可用证书

def test_current_certificate_only_when_latest_unrevoked(client):
    # 从未签发：no_certificate
    make_window(client)
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur == {"window_id": 1, "available": False,
                   "reason": "no_certificate", "certificate": None}

    c1 = issue(client).json()
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is True and cur["reason"] is None
    assert cur["certificate"]["id"] == c1["id"]

    # 撤销最新版：不可用，且不自动退回旧版（此处也没有旧版）
    revoke(client, c1["id"], "依据有误")
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is False and cur["reason"] == "latest_revoked"
    assert cur["certificate"] is None


def test_revoking_latest_does_not_fall_back_to_older(client):
    c1, c2 = seed_issued(client, versions=2)
    # v1、v2 均有效：当前为 v2
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["certificate"]["id"] == c2["id"]

    # 撤销最新版 v2：不自动退回仍有效的 v1
    revoke(client, c2["id"], "新版依据有误")
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is False and cur["reason"] == "latest_revoked"
    assert cur["certificate"] is None

    # v1 本身仍有效且可查（只是不再是“当前可用证书”）
    got1 = client.get(f"/api/v1/certificates/{c1['id']}").json()
    assert got1["revoked"] is False


def test_reissue_after_revoke_uses_incrementing_version_and_becomes_current(client):
    c1 = seed_issued(client)[0]
    revoke(client, c1["id"], "依据有误")
    assert client.get("/api/v1/windows/1/current-certificate").json()["available"] is False

    # 再次签发沿用递增版本号（v2，而不是复用 1），并成为当前可用证书
    c2 = issue(client, note="撤销后重签").json()
    assert c2["version"] == 2 and c2["revoked"] is False
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is True
    assert cur["certificate"]["id"] == c2["id"] and cur["certificate"]["version"] == 2

    # 撤销 v2 后签发 v3：版本继续递增
    revoke(client, c2["id"], "仍有问题")
    c3 = issue(client, note="再次重签").json()
    assert c3["version"] == 3
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is True and cur["certificate"]["id"] == c3["id"]

    # 列表展示全部历史版本及各自撤销状态（按版本升序）
    lst = client.get("/api/v1/windows/1/certificates").json()
    assert [(c["version"], c["revoked"]) for c in lst] == [
        (1, True), (2, True), (3, False)]
    assert lst[0]["revocation"]["reason"] == "依据有误"
    assert lst[1]["revocation"]["reason"] == "仍有问题"
    assert lst[2]["revocation"] is None


def test_current_certificate_missing_window_404(client):
    assert client.get("/api/v1/windows/999/current-certificate").status_code == 404


# ------------------------------------------------------------ 历史可读与报告标示

def test_window_detail_marks_current_and_latest(client):
    c1, c2 = seed_issued(client, versions=2)
    revoke(client, c2["id"], "最新版撤销")

    detail = client.get("/api/v1/windows/1").json()
    # 最新证书元数据仍给出最新版本（用于展示历史），当前可用证书为 None
    assert detail["latest_certificate_version"] == 2
    assert detail["current_certificate"] is None
    assert detail["current_certificate_id"] is None
    # 当前审定仍可读（撤销不影响窗口当前状态计算）
    assert detail["current_evaluation"]["verdict"] in ("pass", "pending", "fail")

    # 再签发 v3 后详情指向 v3
    c3 = issue(client).json()
    detail = client.get("/api/v1/windows/1").json()
    assert detail["current_certificate_id"] == c3["id"]
    assert detail["current_certificate"]["version"] == 3


def test_revision_report_marks_revoked_certificates_but_keeps_snapshot_diff(client):
    # 20 分钟窗口，读数 @00:00/00:10/00:20（间隔恰为 600s，终点读数终止覆盖）
    make_window(client, minutes=20)
    # 三读数：b 超限 -> fail
    client.post("/api/v1/windows/1/readings", json=reading("a", 1, 0, 20.0))
    client.post("/api/v1/windows/1/readings", json=reading("b", 1, 10, 23.0))
    client.post("/api/v1/windows/1/readings", json=reading("c", 1, 20, 20.0))
    c1 = issue(client, note="v1").json()
    assert c1["verdict"] == "fail"

    # 修订读数后签发 v2（pass），随后两张证书都被撤销
    client.post("/api/v1/windows/1/readings", json=reading("b", 2, 10, 20.0))
    c2 = issue(client, note="v2").json()
    assert c2["verdict"] == "pass"
    revoke(client, c1["id"], "旧证依据有误")
    revoke(client, c2["id"], "新证依据也有误")

    # 两证都已撤销，报告仍可生成（只比对历史快照），并标示撤销状态
    rep = client.get(
        "/api/v1/certificates/revision-report"
        f"?old_certificate_id={c1['id']}&new_certificate_id={c2['id']}").json()
    assert rep["old_certificate"]["revoked"] is True
    assert rep["old_certificate"]["status"] == "revoked"
    assert rep["old_certificate"]["revocation"]["reason"] == "旧证依据有误"
    assert rep["old_certificate"]["revocation"]["revoked_at"]
    assert rep["new_certificate"]["revoked"] is True
    assert rep["new_certificate"]["revocation"]["reason"] == "新证依据也有误"
    # 报告内容仍来自不可变快照：fail -> pass
    assert rep["summary"]["verdict_before"] == "fail"
    assert rep["summary"]["verdict_after"] == "pass"
    changed = rep["readings_diff"]["changed"]
    assert len(changed) == 1 and changed[0]["reading_id"] == "b"

    # 撤销新版后旧版撤销状态不受影响；单独撤销一张时另一张标示仍有效
    c3 = issue(client, note="v3").json()
    rep2 = client.get(
        "/api/v1/certificates/revision-report"
        f"?old_certificate_id={c2['id']}&new_certificate_id={c3['id']}").json()
    assert rep2["old_certificate"]["revoked"] is True
    assert rep2["new_certificate"]["revoked"] is False
    assert rep2["new_certificate"]["revocation"] is None


# ------------------------------------------------------------ 并发

def test_concurrent_revoke_and_issue_follow_commit_order(client):
    """撤销与签发并发：BEGIN IMMEDIATE 串行化，两种提交次序下结果都须自洽。

    先有 v1 未撤销：
    - 撤销先提交：v1 已撤销；并发签发生成 v2，v2 未撤销且为当前可用；
    - 签先发提交：得到 v2；撤销命中 v1，当前可用仍为 v2。
    两种次序都不应出现 500，且最终：v1 revoked、共 2 张证书、当前为 v2。
    """
    c1 = seed_issued(client)[0]
    results = {}

    def do_revoke(barrier):
        c = TestClient(client.app)
        barrier.wait()
        r = revoke(c, c1["id"], "并发撤销依据有误")
        results["revoke_code"] = r.status_code
        results["revoke_effect"] = r.json().get("effect")

    def do_issue(barrier):
        c = TestClient(client.app)
        barrier.wait()
        r = issue(c, note="并发签发")
        results["issue_code"] = r.status_code
        if r.status_code == 201:
            results["issue_version"] = r.json()["version"]
            results["issue_revoked"] = r.json()["revoked"]

    barrier = threading.Barrier(2)
    tr = threading.Thread(target=do_revoke, args=(barrier,))
    ti = threading.Thread(target=do_issue, args=(barrier,))
    tr.start(); ti.start(); tr.join(); ti.join()

    assert results["revoke_code"] == 201, results
    assert results["issue_code"] == 201, results
    assert results["issue_version"] == 2
    assert results["issue_revoked"] is False  # 新签发从未被撤销

    # 最终状态与提交次序无关：v1 已撤销、v2 当前可用
    got1 = client.get(f"/api/v1/certificates/{c1['id']}").json()
    assert got1["revoked"] is True
    cur = client.get("/api/v1/windows/1/current-certificate").json()
    assert cur["available"] is True
    assert cur["certificate"]["version"] == 2
    assert cur["certificate"]["revoked"] is False


def test_concurrent_duplicate_revoke_serializes(client):
    """两个线程同时首次撤销同一证书：恰有一次 revoked，另一次幂等返回同记录。"""
    c1 = seed_issued(client)[0]
    results = []

    def do_revoke(barrier):
        c = TestClient(client.app)
        barrier.wait()
        r = revoke(c, c1["id"], "同一依据原因")
        results.append((r.status_code, r.json().get("effect"),
                        r.json().get("revocation", {}).get("revoked_at")))

    barrier = threading.Barrier(2)
    threads = [threading.Thread(target=do_revoke, args=(barrier,)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(code == 201 for code, _, _ in results), results
    effects = sorted(effect for _, effect, _ in results)
    assert effects == ["idempotent", "revoked"], results
    # 两次返回的是同一条撤销记录
    assert results[0][2] == results[1][2]

    from app.database import connect
    conn = connect()
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM certificate_revocations WHERE certificate_id=?",
            (c1["id"],),
        ).fetchone()["n"]
        assert n == 1
    finally:
        conn.close()
