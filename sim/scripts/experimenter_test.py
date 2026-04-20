"""Playwright verification for experimenter dashboard and drone ablation toggle."""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = "http://localhost:5175/"
SHOTS = Path("/tmp/sim_screenshots")
SHOTS.mkdir(exist_ok=True)


def wait_for_app(page: Page) -> None:
    page.goto(URL)
    page.wait_for_selector("#scenario-select", state="visible", timeout=120_000)
    page.wait_for_selector("#btn-run:not([disabled])", state="visible", timeout=180_000)


def switch(page: Page, scenario: str, view: str) -> None:
    page.select_option("#scenario-select", scenario)
    page.wait_for_selector("#btn-run:not([disabled])", timeout=120_000)
    page.select_option("#mode-select", view)
    if view == "experimenter":
        page.wait_for_selector("#exp-summary-table", timeout=10_000)
        page.wait_for_selector("#exp-episodes", timeout=10_000)


def set_exp_params(page: Page, *, episodes: int, steps: int, batch: int = 5,
                   ablation: bool | None = None) -> None:
    page.fill("#exp-episodes", str(episodes))
    page.fill("#exp-steps", str(steps))
    page.fill("#exp-batch", str(batch))
    if ablation is not None:
        cb = page.locator("#exp-ablation")
        if cb.count() > 0:
            checked = cb.is_checked()
            if checked != ablation:
                cb.click()


def parse_pct(s: str) -> float | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*%", s or "")
    return float(m.group(1)) if m else None


def parse_num(s: str) -> float | None:
    m = re.search(r"(-?\d+(?:\.\d+)?)", s or "")
    return float(m.group(1)) if m else None


def parse_leaderboard(page: Page) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict] = {}
    for row in page.locator(".exp-row").all():
        agent = (row.locator(".exp-agent").text_content() or "").strip()
        if not agent:
            continue
        record: dict[str, float | None] = {}
        for m in row.locator(".exp-metric").all():
            text = (m.text_content() or "").strip()
            if ":" in text:
                key, _, val = text.partition(":")
                key = key.strip().lower()
                val = val.strip()
                record[key] = parse_pct(val) if "%" in val else parse_num(val)
        out[agent] = record
    return out


def run_benchmark(page: Page, *, timeout_ms: int = 900_000) -> str:
    page.click("#btn-run")
    deadline = time.time() + timeout_ms / 1000
    last_status = ""
    while time.time() < deadline:
        last_status = page.locator("#exp-status-panel").text_content() or ""
        if "complete" in last_status.lower() and "Run" not in last_status:
            return last_status
        if "complete" in last_status.lower():
            return last_status
        time.sleep(0.4)
    raise TimeoutError(f"Benchmark did not finish; last status: {last_status!r}")


def show(label: str, board: dict[str, dict]) -> None:
    print(f"  --- {label} ---")
    for agent, rec in board.items():
        print(f"    {agent:22s} {rec}")


def clear_benchmark(page: Page) -> None:
    page.click("#btn-reset")
    deadline = time.time() + 8
    while time.time() < deadline:
        board = parse_leaderboard(page)
        eps = [v.get("episodes", 0) or 0 for v in board.values()]
        if not eps or max(eps) == 0:
            return
        time.sleep(0.3)


def get(rec: dict, *keys: str) -> float | None:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return None


def find_agent(board: dict, needle: str) -> dict | None:
    for k, v in board.items():
        if needle.lower() in k.lower():
            return v
    return None


def test_tmaze_experimenter(page: Page) -> None:
    print("\n=== T-MAZE experimenter ===")
    switch(page, "tmaze", "experimenter")
    set_exp_params(page, episodes=20, steps=32, batch=4)
    page.screenshot(path=str(SHOTS / "exp_01_tmaze_before.png"))
    status = run_benchmark(page, timeout_ms=300_000)
    print(f"  status: {status!r}")
    page.screenshot(path=str(SHOTS / "exp_02_tmaze_after.png"))
    board = parse_leaderboard(page)
    show("T-maze leaderboard", board)
    assert len(board) >= 5, f"Expected ≥5 agents, got {len(board)}"
    rand = find_agent(board, "random") or {}
    greedy = find_agent(board, "greedy") or {}
    combined = find_agent(board, "combined") or {}
    al = find_agent(board, "learning") or {}
    rand_acc = get(rand, "reward rate", "accuracy")
    greedy_acc = get(greedy, "reward rate", "accuracy")
    combined_acc = get(combined, "reward rate", "accuracy")
    greedy_cue = get(greedy, "cue")
    al_cue = get(al, "cue")
    rand_cue = get(rand, "cue")
    print(f"  [check] random reward ~50% (got {rand_acc}%)")
    print(f"  [check] greedy reward >=70% (got {greedy_acc}%)")
    print(f"  [check] combined reward >=70% (got {combined_acc}%)")
    print(f"  [paper] greedy cue ~100% (got {greedy_cue}%)")
    print(f"  [paper] AL cue ~85-95% (got {al_cue}%)")
    print(f"  [paper] random cue ~33% (got {rand_cue}%)")
    assert 30 <= (rand_acc or 0) <= 70, f"random reward not ~50%: {rand_acc}"
    assert (greedy_acc or 0) >= 70, f"greedy reward too low: {greedy_acc}"
    assert (combined_acc or 0) >= 70, f"combined reward too low: {combined_acc}"
    assert (greedy_cue or 0) >= 95, f"greedy should always cue: {greedy_cue}"
    assert 20 <= (rand_cue or 0) <= 50, f"random cue not ~33%: {rand_cue}"


def test_grid_experimenter(page: Page) -> None:
    print("\n=== GRID MAZE experimenter ===")
    switch(page, "grid_maze", "experimenter")
    assert page.locator("#exp-ablation").count() == 0, "grid_maze should not show ablation toggle"
    set_exp_params(page, episodes=15, steps=60, batch=3)
    page.screenshot(path=str(SHOTS / "exp_03_grid_before.png"))
    status = run_benchmark(page, timeout_ms=300_000)
    print(f"  status: {status!r}")
    page.screenshot(path=str(SHOTS / "exp_04_grid_after.png"))
    board = parse_leaderboard(page)
    show("Grid leaderboard", board)
    assert len(board) >= 5
    rand_acc = get(find_agent(board, "random") or {}, "success rate", "reward rate", "accuracy")
    combined_acc = get(find_agent(board, "combined") or {}, "success rate", "reward rate", "accuracy")
    print(f"  [check] random < 75% (got {rand_acc}%)")
    print(f"  [check] combined >= 65% (got {combined_acc}%)")
    assert (rand_acc if rand_acc is not None else 100) < 75
    assert (combined_acc or 0) >= 65


def test_drone_ablation(page: Page) -> None:
    print("\n=== DRONE SEARCH ablation toggle ===")
    switch(page, "drone_search", "experimenter")
    assert page.locator("#exp-ablation").count() == 1, "Expected ablation checkbox for drone_search"

    set_exp_params(page, episodes=10, steps=200, batch=5, ablation=False)
    page.screenshot(path=str(SHOTS / "exp_05_drone_full_before.png"))
    status_full = run_benchmark(page, timeout_ms=600_000)
    print(f"  full status: {status_full!r}")
    assert "ablation" not in status_full.lower()
    page.screenshot(path=str(SHOTS / "exp_06_drone_full_after.png"))
    board_full = parse_leaderboard(page)
    show("Drone FULL EFE", board_full)

    clear_benchmark(page)
    set_exp_params(page, episodes=10, steps=200, batch=5, ablation=True)
    page.screenshot(path=str(SHOTS / "exp_07_drone_ablated_before.png"))
    status_abl = run_benchmark(page, timeout_ms=600_000)
    print(f"  ablated status: {status_abl!r}")
    page.screenshot(path=str(SHOTS / "exp_08_drone_ablated_after.png"))
    board_abl = parse_leaderboard(page)
    show("Drone ABLATED", board_abl)

    keys = ("reward rate", "accuracy", "success rate", "success")
    g_full = get(find_agent(board_full, "greedy") or {}, *keys)
    g_abl = get(find_agent(board_abl, "greedy") or {}, *keys)
    c_full = get(find_agent(board_full, "combined") or {}, *keys)
    c_abl = get(find_agent(board_abl, "combined") or {}, *keys)
    print(f"  [info] greedy   full={g_full}% ablated={g_abl}%")
    print(f"  [info] combined full={c_full}% ablated={c_abl}%")
    assert c_full is not None and c_abl is not None


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"[console.{m.type}] {m.text}") if m.type == "error" else None)

        wait_for_app(page)
        try:
            test_tmaze_experimenter(page)
            test_grid_experimenter(page)
            test_drone_ablation(page)
        except Exception as exc:
            print(f"\nFAILED: {exc}")
            page.screenshot(path=str(SHOTS / "exp_failure.png"))
            print(f"errors: {errors[:8]}")
            browser.close()
            return 1
        print(f"\n=== ERRORS ({len(errors)}) ===")
        for e in errors[:10]:
            print(f"  {e}")
        browser.close()
    print(f"\nScreenshots: {SHOTS}/")
    print("\nALL EXPERIMENTER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
