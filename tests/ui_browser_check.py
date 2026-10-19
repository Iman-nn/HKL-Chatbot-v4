"""Optional isolated Chromium checks. Requires Playwright and its Chromium binary.

Run from the project root while the local preview is listening on port 5050.
"""

from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright

    output = Path(__file__).resolve().parent.parent / "artifacts"
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width":1440,"height":1000},device_scale_factor=1)
            errors=[]
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.goto("http://127.0.0.1:5050/",wait_until="networkidle")
            page.screenshot(path=str(output/"ui-desktop.png"),full_page=True,animations="disabled")
            assert page.locator("#user-input").get_attribute("placeholder") == "Ask HKL Guide a question..."
            assert page.locator("#welcome").is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            page.locator("#toggle-dark").click()
            page.screenshot(path=str(output/"ui-dark.png"),full_page=True,animations="disabled")
            page.locator("#browse-faqs").click()
            page.locator(".explore-item").first.wait_for()
            assert page.locator("#explore-dialog").is_visible()
            page.locator("#explore-content .explore-item").filter(has_text="Library").first.click()
            page.locator("#explore-content .explore-item").first.wait_for()
            page.locator("#explore-content .explore-item").first.click()
            page.locator(".message.bot").first.wait_for()
            assert page.locator("#welcome").is_hidden()
            page.screenshot(path=str(output/"ui-conversation.png"),full_page=True,animations="disabled")
            page.locator("#clear-chat").click()
            page.locator("#welcome").wait_for()
            page.locator("#browse-locations").click()
            page.locator("#location-query").fill("Pharmacy")
            page.locator("#location-form button").click()
            page.locator("#explore-content .explore-item").first.wait_for()
            assert "Pharmacy" in page.locator("#explore-content").inner_text()
            page.screenshot(path=str(output/"ui-locations.png"),full_page=True,animations="disabled")
            page.locator("#close-explore").click()
            mobile=browser.new_page(viewport={"width":390,"height":844},is_mobile=True,device_scale_factor=1)
            mobile.goto("http://127.0.0.1:5050/",wait_until="networkidle")
            assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            mobile.screenshot(path=str(output/"ui-mobile.png"),full_page=True,animations="disabled")
            mobile.set_viewport_size({"width":320,"height":740})
            assert mobile.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            mobile.screenshot(path=str(output/"ui-mobile-small.png"),full_page=True,animations="disabled")
            mobile.locator("#browse-locations").click()
            bounds=mobile.locator("#explore-dialog").bounding_box()
            assert bounds["x"] >= 0 and bounds["x"]+bounds["width"] <= 321
            mobile.locator("#close-explore").click()
            assert not errors, errors
            print("PASS: isolated Chromium visual captures, desktop/mobile width checks, themes, FAQ -> chat -> reset, locations, no page errors")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
