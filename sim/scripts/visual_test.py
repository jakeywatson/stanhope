"""Visual regression tests for all three scenarios.
Screenshots at key stages: initial load, after steps, after experiment.
Also validates DOM state (panel values, step counters).
"""
from playwright.sync_api import sync_playwright
import os, json

SCREENSHOTS = "/tmp/sim_screenshots"
os.makedirs(SCREENSHOTS, exist_ok=True)

URL = "http://localhost:5175/"

def screenshot(page, name):
    path = f"{SCREENSHOTS}/{name}.png"
    page.screenshot(path=path, full_page=False)
    print(f"  📸 {name}.png")
    return path

def get_panel_state(page, ids):
    """Read text content from a list of element IDs."""
    out = {}
    for eid in ids:
        el = page.query_selector(f"#{eid}")
        out[eid] = el.text_content().strip() if el else "MISSING"
    return out

def test_tmaze(page):
    print("\n=== T-MAZE ===")
    # Should already be default scenario
    page.wait_for_timeout(500)
    screenshot(page, "01_tmaze_initial")

    # Check panel elements exist
    panel_html = page.inner_html("#sidepanel")
    has_context = "Hidden Context" in panel_html or "context" in panel_html.lower()
    has_policy = "Policy" in panel_html
    has_efe = "Free Energy" in panel_html or "EFE" in panel_html.upper()
    print(f"  Panel sections — Context: {has_context}, Policy: {has_policy}, EFE: {has_efe}")

    # Step once
    page.click("#btn-step")
    page.wait_for_timeout(2500)  # T-maze animates waypoints
    screenshot(page, "02_tmaze_after_step")
    step = page.text_content("#trial-num")
    print(f"  Step counter after 1 step: '{step}'")

    # Run 10 more
    page.click("#btn-run")
    page.wait_for_timeout(3000)
    screenshot(page, "03_tmaze_after_run")
    step = page.text_content("#trial-num")
    print(f"  Step counter after Run: '{step}'")

    # Check some panel values updated
    # Look for any non-zero bar widths
    bars = page.query_selector_all(".bar-fill")
    nonzero_bars = 0
    for bar in bars:
        w = bar.evaluate("el => el.style.width")
        if w and w != "0%" and w != "0px":
            nonzero_bars += 1
    print(f"  Non-zero bars: {nonzero_bars}/{len(bars)}")

    efe_fills = page.query_selector_all(".efe-fill")
    nonzero_efe = 0
    for ef in efe_fills:
        w = ef.evaluate("el => el.style.width")
        if w and w != "0%" and w != "0px":
            nonzero_efe += 1
    print(f"  Non-zero EFE bars: {nonzero_efe}/{len(efe_fills)}")

    # Check reward chart has content
    canvas = page.query_selector("#reward-chart")
    has_canvas = canvas is not None
    print(f"  Reward chart canvas: {has_canvas}")

def test_grid_maze(page):
    print("\n=== GRID MAZE ===")
    page.select_option("#scenario-select", "grid_maze")
    page.wait_for_timeout(1000)
    screenshot(page, "04_grid_initial")

    # Check panel
    panel_html = page.inner_html("#sidepanel")
    has_nav = "Navigation" in panel_html
    has_policy = "Policy" in panel_html
    has_efe = "EFE" in panel_html
    print(f"  Panel sections — Navigation: {has_nav}, Policy: {has_policy}, EFE: {has_efe}")

    explored_el = page.query_selector("#val-explored")
    explored = explored_el.text_content() if explored_el else "MISSING"
    print(f"  Initial explored: '{explored}'")

    # Step 5 times
    for _ in range(5):
        page.click("#btn-step")
        page.wait_for_timeout(400)
    screenshot(page, "05_grid_after_5_steps")

    explored = page.text_content("#val-explored") if page.query_selector("#val-explored") else "MISSING"
    steps = page.text_content("#val-steps") if page.query_selector("#val-steps") else "MISSING"
    trial = page.text_content("#trial-num")
    print(f"  After 5 steps: explored='{explored}', steps='{steps}', counter='{trial}'")

    # Check policy bars
    for dir in ["north", "south", "east", "west"]:
        el = page.query_selector(f"#pol-{dir}")
        val_el = page.query_selector(f"#pol-val-{dir}")
        w = el.evaluate("el => el.style.width") if el else "MISSING"
        v = val_el.text_content() if val_el else "MISSING"
        print(f"  Policy {dir}: bar={w}, val={v}")

    # Run burst
    page.click("#btn-run")
    page.wait_for_timeout(5000)
    screenshot(page, "06_grid_after_run")
    explored = page.text_content("#val-explored") if page.query_selector("#val-explored") else "MISSING"
    trial = page.text_content("#trial-num")
    print(f"  After Run: explored='{explored}', counter='{trial}'")

    # Run experiment
    page.click("#btn-experiment")
    page.wait_for_timeout(30000)
    screenshot(page, "07_grid_after_experiment")
    explored = page.text_content("#val-explored") if page.query_selector("#val-explored") else "MISSING"
    goal_status = page.text_content("#goal-status") if page.query_selector("#goal-status") else "MISSING"
    trial = page.text_content("#trial-num")
    print(f"  After Experiment: explored='{explored}', goal='{goal_status}', counter='{trial}'")

    # Check for visible 3D content (canvas should have non-black pixels — check size)
    canvas = page.query_selector("#viewport canvas")
    if canvas:
        box = canvas.bounding_box()
        print(f"  3D canvas size: {box['width']}x{box['height']}")
    else:
        print(f"  ⚠️ No canvas in viewport!")

def test_drone(page):
    print("\n=== DRONE SEARCH ===")
    page.select_option("#scenario-select", "drone_search")
    page.wait_for_timeout(1000)
    screenshot(page, "08_drone_initial")

    panel_html = page.inner_html("#sidepanel")
    has_status = "Drone Status" in panel_html
    has_policy = "Policy" in panel_html
    has_efe = "EFE" in panel_html
    print(f"  Panel sections — Status: {has_status}, Policy: {has_policy}, EFE: {has_efe}")

    coverage = page.text_content("#val-coverage") if page.query_selector("#val-coverage") else "MISSING"
    targets = page.text_content("#val-targets") if page.query_selector("#val-targets") else "MISSING"
    alt = page.text_content("#val-alt") if page.query_selector("#val-alt") else "MISSING"
    print(f"  Initial: coverage='{coverage}', targets='{targets}', alt='{alt}'")

    # Step 5 times
    for _ in range(5):
        page.click("#btn-step")
        page.wait_for_timeout(400)
    screenshot(page, "09_drone_after_5_steps")

    coverage = page.text_content("#val-coverage") if page.query_selector("#val-coverage") else "MISSING"
    targets = page.text_content("#val-targets") if page.query_selector("#val-targets") else "MISSING"
    alt = page.text_content("#val-alt") if page.query_selector("#val-alt") else "MISSING"
    trial = page.text_content("#trial-num")
    print(f"  After 5 steps: coverage='{coverage}', targets='{targets}', alt='{alt}', counter='{trial}'")

    # Check chosen action updates
    chosen = page.text_content("#chosen-action") if page.query_selector("#chosen-action") else "MISSING"
    print(f"  Chosen action: '{chosen}'")

    # Run experiment
    page.click("#btn-experiment")
    page.wait_for_timeout(20000)
    screenshot(page, "10_drone_after_experiment")

    coverage = page.text_content("#val-coverage") if page.query_selector("#val-coverage") else "MISSING"
    targets = page.text_content("#val-targets") if page.query_selector("#val-targets") else "MISSING"
    trial = page.text_content("#trial-num")
    scan_r = page.text_content("#scan-radius") if page.query_selector("#scan-radius") else "MISSING"
    print(f"  After Experiment: coverage='{coverage}', targets='{targets}', counter='{trial}', radius='{scan_r}'")

    # Policy bars check
    for dir in ["north", "south", "east", "west", "up", "down"]:
        el = page.query_selector(f"#pol-val-{dir}")
        v = el.text_content() if el else "MISSING"
        if v != "0.17":
            print(f"  Policy {dir} updated: {v}")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        errors = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        print("Loading app...")
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("#loading.hidden", timeout=120000)
        print("App loaded!")

        test_tmaze(page)
        test_grid_maze(page)
        test_drone(page)

        print(f"\n=== ERRORS ({len(errors)}) ===")
        for e in errors:
            print(e[:500])

        print(f"\nScreenshots saved to {SCREENSHOTS}/")
        browser.close()

if __name__ == "__main__":
    main()
