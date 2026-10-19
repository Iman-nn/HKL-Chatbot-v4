"""Optional isolated Chromium QA; start an AI-disabled local preview on 5051.

Requires Playwright and Chromium for QA only. Provider responses are mocked in
the browser; no API key or external AI call is used by this check.
"""

from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright

    output = Path(__file__).resolve().parent.parent / "artifacts"
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:5051/", wait_until="networkidle")
            page.locator("#user-input").fill("how to contact hospital?")
            page.locator("#send-btn").click()
            answer = page.locator(".message.bot").last
            answer.wait_for()
            assert answer.locator("strong").first.inner_text() == "Main switchboard:"
            assert answer.locator('a[href="tel:0326155555"]').count() == 1
            assert "Checked 2026-10-19" in answer.locator(".answer-source").inner_text()
            assert page.locator("#suggestions button").count() == 0
            assert "**" not in answer.inner_text()
            page.screenshot(path=str(output / "response-desktop.png"), full_page=True)
            page.locator("#toggle-dark").click()
            page.screenshot(path=str(output / "response-dark.png"), full_page=True, animations="disabled")
            for width in (390, 320):
                page.set_viewport_size({"width": width, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                bounds = answer.bounding_box()
                assert bounds["x"] >= 0 and bounds["x"] + bounds["width"] <= width
            page.screenshot(path=str(output / "response-mobile.png"), full_page=True)
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator("#toggle-dark").click()
            # The real HTTP contact answer in Malay must bypass the provider too.
            page.locator("#user-input").fill("Nombor telefon HKL?")
            page.locator("#send-btn").click()
            page.locator(".message.bot").nth(1).wait_for()
            assert "Telefon utama:" in page.locator(".message.bot").last.inner_text()
            page.route("**/ask/*", lambda route: route.fulfill(json={
                "response": "**Next step:** Ask the hospital for help.\n\n- Visit [HKL](https://hkl.moh.gov.my/)\n- Call 03-2615 5555\n\n<img src=x onerror=alert(1)> [Unsafe](javascript:alert)",
                "source": "external_ai", "suggestions": [],
            }))
            page.locator("#user-input").fill("Show a formatted answer")
            page.locator("#send-btn").click()
            page.locator(".message.bot").nth(2).wait_for()
            ai = page.locator(".message.bot").last
            assert ai.locator(".answer-badge").inner_text() == "AI-generated · Please verify"
            assert ai.locator("li").count() == 2
            assert ai.locator("img, script, a[href^='javascript:']").count() == 0
            assert ai.locator(".answer-source").count() == 0
            page.screenshot(path=str(output / "response-ai.png"), full_page=True)
            assert not errors, errors
            print("PASS: real contact EN/MS, source footer, phone link, no irrelevant suggestions, safe AI formatting, desktop/dark/mobile, no JS errors")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
