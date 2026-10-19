"""Optional isolated Chromium layout QA; local preview on 5052, no AI calls.

Requires Playwright and Chromium for QA only. All chat API responses are mocked.
Viewport shrinking exercises the compact layout; it does not replace testing
an actual iOS/Android software keyboard or a physical device.
"""

from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright, expect

    output = Path(__file__).resolve().parent.parent / "artifacts"
    output.mkdir(exist_ok=True)
    sizes = [(320, 740), (360, 780), (390, 844), (600, 900), (768, 1024),
             (820, 1180), (960, 720), (1024, 768), (1280, 720), (1440, 900),
             (1920, 1080), (2560, 1440), (844, 390), (667, 375), (568, 320)]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/start_conversation", lambda route: route.fulfill(json={
                "conversation_id": "qa", "conversation_token": "not-a-real-token"}))
            answer = "**Hospital guide:** Here are the next steps.\n\n" + "\n\n".join(
                "Ask hospital staff if you need help finding your clinic. " * 5 for _ in range(8))
            answer += "\n\nhttps://hkl.moh.gov.my/" + "a" * 180
            page.route("**/ask/*", lambda route: route.fulfill(json={
                "response": answer, "source": "external_ai", "suggestions": [
                    "How do I find the hospital pharmacy?", "What are the library opening hours?",
                    "Where can I ask about hospital services?"
                ]}))
            page.goto("http://127.0.0.1:5052/", wait_until="networkidle")
            # Include the optional disclosure in layout checks without enabling
            # any external provider or exposing a real API key to the test.
            page.evaluate("""() => {
                const notice = document.createElement('p');
                notice.className = 'external-notice';
                notice.textContent = 'Unmatched questions may be sent to an external AI provider. Please verify AI-generated answers.';
                document.querySelector('.composer-section').append(notice);
            }""")

            def layout(width, height):
                page.set_viewport_size({"width": width, "height": height})
                # Wait for the visualViewport resize handler to finish.
                page.wait_for_function("() => Math.abs(document.querySelector('.app-shell').getBoundingClientRect().height - window.innerHeight) < 2")
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (width, height, "page overflow")
                assert page.evaluate("document.documentElement.scrollHeight <= window.innerHeight + 1"), (width, height, "page height")
                for selector in ("#chat-form", "#send-btn", "#user-input", ".composer-section"):
                    box = page.locator(selector).bounding_box()
                    assert box["x"] >= -1 and box["x"] + box["width"] <= width + 1, (width, height, selector, box)
                    assert box["y"] >= -1 and box["y"] + box["height"] <= height + 1, (width, height, selector, box)
                assert page.locator("#conversation-area").bounding_box()["height"] >= 70, (width, height, "no reading space")
                assert page.locator("#send-btn").bounding_box()["width"] >= 44

            for width, height in sizes:
                layout(width, height)
            page.locator("#user-input").fill("A long question " * 100)
            short_height = page.locator("#user-input").bounding_box()["height"]
            assert short_height > 44
            assert short_height <= 161
            page.locator("#user-input").fill("Show a responsive answer")
            assert page.locator("#user-input").bounding_box()["height"] < short_height
            page.locator("#send-btn").click()
            expect(page.locator(".message.bot")).to_have_count(1)
            for width, height in sizes:
                layout(width, height)
                assert page.evaluate("document.querySelector('#conversation-area').scrollWidth <= document.querySelector('#conversation-area').clientWidth"), (width, height, "chat overflow")
            layout(1440, 900)
            page.locator("#conversation-area").evaluate("area => area.scrollTop = 0")
            expect(page.locator("#jump-latest")).to_be_visible()
            page.locator("#jump-latest").click()
            expect(page.locator("#jump-latest")).to_be_hidden()
            page.route("**/history/*", lambda route: route.fulfill(status=204))
            page.locator("#clear-chat").click()
            expect(page.locator("#welcome")).to_be_visible()
            page.route("**/ask/*", lambda route: route.fulfill(json={
                "response": "**Find your clinic or facility**\n\nYou can search for a destination using Hospital directions.\n\n1. Open **Hospital directions**.\n2. Enter a clinic or facility name, such as Pharmacy.\n3. Choose a result to see the available directions.\n\nIf you need help along the way, ask hospital staff.",
                "source": "external_ai", "suggestions": ["Where is Pharmacy?", "What are the library opening hours?"]}))
            page.locator("#user-input").fill("How can I find my clinic?")
            page.locator("#send-btn").click()
            expect(page.locator(".message.bot")).to_have_count(1)
            page.locator("#toggle-dark").click()
            # Capture a readable sample after exercising the stress case.
            for name, width, height in [("wide", 1920, 1080), ("tablet", 768, 1024),
                                        ("mobile", 390, 844), ("landscape", 844, 390)]:
                layout(width, height)
                page.screenshot(path=str(output / f"responsive-{name}.png"), animations="disabled")
            # Shrink the viewport as a keyboard would, then restore it.
            layout(390, 350)
            page.screenshot(path=str(output / "responsive-compact.png"), animations="disabled")
            layout(390, 844)
            for button, dialog in [("#browse-faqs", "#explore-dialog"), ("#privacy-inline", "#privacy-dialog")]:
                page.locator(button).click()
                for width, height in [(320, 740), (844, 390)]:
                    layout(width, height)
                    box = page.locator(dialog).bounding_box()
                    assert box["x"] >= 0 and box["x"] + box["width"] <= width + 1
                    assert box["y"] >= 0 and box["y"] + box["height"] <= height + 1
                page.keyboard.press("Escape")
            # Exercise the visualViewport-only keyboard path (Safari-style),
            # where innerHeight remains tall while the visible area shrinks.
            keyboard = browser.new_page(viewport={"width": 390, "height": 844})
            keyboard.on("pageerror", lambda error: errors.append(str(error)))
            keyboard.add_init_script("""
                const viewport = new EventTarget();
                Object.assign(viewport, {height: 844, offsetTop: 0, scale: 1});
                Object.defineProperty(window, 'visualViewport', {value: viewport});
            """)
            keyboard.goto("http://127.0.0.1:5052/", wait_until="networkidle")
            keyboard.locator("#user-input").focus()
            keyboard.evaluate("() => { visualViewport.height = 350; visualViewport.dispatchEvent(new Event('resize')); }")
            assert keyboard.locator("html").get_attribute("class") == "keyboard-open"
            assert keyboard.locator("#chat-form").bounding_box()["y"] + keyboard.locator("#chat-form").bounding_box()["height"] <= 350
            keyboard.locator("#user-input").blur()
            keyboard.locator("#browse-locations").click()
            keyboard.locator("#location-query").focus()
            keyboard.evaluate("() => visualViewport.dispatchEvent(new Event('resize'))")
            box = keyboard.locator("#explore-dialog").bounding_box()
            assert box["y"] >= 0 and box["y"] + box["height"] <= 350
            keyboard.keyboard.press("Escape")
            keyboard.evaluate("() => { visualViewport.height = 844; visualViewport.dispatchEvent(new Event('resize')); }")
            assert "keyboard-open" not in (keyboard.locator("html").get_attribute("class") or "")
            keyboard.close()
            assert not errors, errors
            print(f"PASS: {len(sizes)} viewport sizes, welcome/chat, long URLs, auto-growing input, touch controls, latest-message navigation, dialogs, compact viewport, no JS errors")
        finally:
            browser.close()


if __name__ == "__main__":
    main()
