// Run the real client with a small DOM test double and Node's built-in test runner.
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");
const assert = require("node:assert/strict");

class Element {
  constructor(tag = "div") {
    this.tagName = tag;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.value = "";
    this.textContent = "";
    this.disabled = false;
    this.className = "";
    this.hidden = false;
    this.open = false;
    const classes = new Set();
    this.classList = { toggle(name) {
      if (classes.has(name)) { classes.delete(name); return false; }
      classes.add(name); return true;
    } };
  }
  set innerHTML(_) { throw new Error("Unsafe HTML rendering"); }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(""); }
  get firstElementChild() { return this.children[0]; }
  append(node) { node.parent = this; this.children.push(node); }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  replaceChildren() { this.children = []; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  focus() {}
}

function harness(fetch) {
  const elements = new Map();
  const document = {
    body: new Element("body"),
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, new Element());
      return elements.get(selector);
    },
    querySelectorAll(selector) {
      if (selector !== "[data-question]") return [];
      const question = document.querySelector("#library-shortcut");
      question.setAttribute("data-question", "What are the library opening hours?");
      return [question];
    },
    createElement: tag => new Element(tag),
    createTextNode(text) { const node = new Element("#text"); node.textContent = text; return node; },
  };
  const context = vm.createContext({ document, window: {}, fetch, URL, AbortController, setTimeout, clearTimeout });
  vm.runInContext(fs.readFileSync("static/chat.js", "utf8"), context);
  return { context, elements, run: code => vm.runInContext(code, context) };
}

const success = data => ({ ok: true, status: 200, json: async () => data });

function descendants(element, tag) {
  return element.children.flatMap(child => [
    ...(child.tagName === tag ? [child] : []), ...descendants(child, tag),
  ]);
}

test("bot formatting creates semantic paragraphs, bold labels, lists and actionable links", () => {
  const state = harness();
  state.run('appendMessage("**Phone:** 03-2615 5555\\n**Website:** https://hkl.moh.gov.my/.\\n\\n- First step\\n- Second step\\n\\nEmail: pro.hkl@moh.gov.my");');
  const bubble = state.elements.get("#chatbox").children[0];
  assert.equal(descendants(bubble, "strong")[0].textContent, "Phone:");
  assert.equal(descendants(bubble, "ul").length, 1);
  assert.equal(descendants(bubble, "li").length, 2);
  const links = descendants(bubble, "a");
  assert.equal(links[0].attributes.href, "tel:0326155555");
  assert.equal(links[1].attributes.href, "https://hkl.moh.gov.my/");
  assert.equal(links[1].attributes.rel, "noopener noreferrer");
  assert.equal(links[2].attributes.href, "mailto:pro.hkl@moh.gov.my");
  assert.equal(bubble.textContent.includes("**"), false);
  state.run('appendMessage("Call +60 3-2615 5555 ext. 7016");');
  assert.equal(descendants(state.elements.get("#chatbox").children[1], "a")[0].attributes.href,
    "tel:+60326155555;ext=7016");
});

test("untrusted Markdown and HTML cannot create executable links or elements", () => {
  const state = harness();
  state.run('appendMessage("<img src=x onerror=alert(1)> **Safe** [Bad](javascript:alert) [Data](data:text/html,test) [Hidden](https://user:password@example.com) [Good](https://hkl.moh.gov.my/)");');
  const bubble = state.elements.get("#chatbox").children[0];
  assert.equal(descendants(bubble, "img").length, 0);
  assert.equal(descendants(bubble, "a").length, 1);
  assert.equal(descendants(bubble, "a")[0].attributes.href, "https://hkl.moh.gov.my/");
  assert.match(bubble.textContent, /<img src=x onerror=alert\(1\)>/);
  for (const url of ["javascript:alert(1)", "data:text/html,test", "//evil.test", "https://user:pass@evil.test", "https://hkl.moh.gov.my/\n", "https:evil.test"]) {
    assert.equal(state.run(`safeLink(${JSON.stringify(url)})`), null);
  }
});

test("AI disclosure is separate and only curated hospital references become source footers", () => {
  const state = harness();
  state.run('appendMessage("**Answer:** A short reply.", "bot", {source:"external_ai", references:[{url:"https://hkl.moh.gov.my/"}]});');
  const ai = state.elements.get("#chatbox").children[0];
  assert.equal(ai.children[0].className, "answer-badge");
  assert.match(ai.children[0].textContent, /AI-generated/);
  assert.equal(ai.children.length, 2);
  state.run('appendMessage("Phone: 03-2615 5555", "bot", {source:"official_contact", references:[{title:"HKL official website",url:"https://hkl.moh.gov.my/en/",checked_on:"2026-10-19"},{url:"https://evil.test"},{url:"javascript:alert(1)"}]});');
  const contact = state.elements.get("#chatbox").children[1];
  assert.equal(contact.children[1].className, "answer-source");
  assert.match(contact.children[1].textContent, /Checked 2026-10-19/);
  assert.equal(contact.children.length, 2);
});

test("user, provider, and suggestion payloads remain literal text", () => {
  const state = harness();
  state.run('appendMessage("<img src=x onerror=alert(1)>", "user"); showSuggestions(["<svg onload=alert(1)>"]);');
  const bubble = state.elements.get("#chatbox").children.at(-1);
  assert.equal(bubble.textContent, "<img src=x onerror=alert(1)>");
  const suggestion = state.elements.get("#suggestions").children[0];
  assert.equal(suggestion.textContent, "<svg onload=alert(1)>");
  assert.equal(suggestion.attributes.onclick, undefined);
  suggestion.listeners.click();
  assert.equal(state.elements.get("#user-input").value, "<svg onload=alert(1)>");
});

test("lazy initialization precedes ask and includes the bearer token", async () => {
  const calls = [];
  const state = harness(async (path, options) => {
    calls.push({ path, options });
    return success(path === "/start_conversation"
      ? { conversation_id: "id", conversation_token: "access-token" }
      : { response: "Answer", suggestions: [] });
  });
  state.elements.get("#user-input").value = "Question";
  await state.run("sendMessage()");
  assert.deepEqual(calls.map(call => call.path), ["/start_conversation", "/ask/id"]);
  assert.equal(calls[1].options.headers.Authorization, "Bearer access-token");
  assert.equal(state.elements.get("#send-btn").disabled, false);
  assert.equal(state.elements.get("#chatbox").children.at(-1).textContent, "Answer");
});

test("simultaneous sends create one conversation and one ask", async () => {
  let calls = 0;
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const state = harness(async path => {
    calls++;
    if (path === "/start_conversation") {
      await pending;
      return success({ conversation_id: "id", conversation_token: "token" });
    }
    return success({ response: "Answer", suggestions: [] });
  });
  state.elements.get("#user-input").value = "Question";
  const first = state.run("sendMessage()");
  await state.run("sendMessage()");
  assert.equal(calls, 1);
  release();
  await first;
  assert.equal(calls, 2);
});

test("expired sessions preserve input and allow a new session on the next send", async () => {
  const state = harness(async path => path === "/start_conversation"
    ? success({ conversation_id: "id", conversation_token: "token" })
    : { ok: false, status: 404, json: async () => ({ error: "Expired" }) });
  state.elements.get("#user-input").value = "My question";
  await state.run("sendMessage()");
  assert.equal(state.run("conversation"), null);
  assert.equal(state.elements.get("#user-input").value, "My question");
  assert.equal(state.elements.get("#send-btn").disabled, false);
});

test("message history is bounded in the DOM", () => {
  const state = harness();
  state.run('for (let i = 0; i < 200; i++) appendMessage("Message " + i);');
  assert.equal(state.elements.get("#chatbox").children.length, 150);
});

test("FAQ explorer uses real API categories and asks the selected question", async () => {
  const paths = [];
  const state = harness(async (path, options) => {
    paths.push(path);
    if (path === "/faq_classes") return success({ classes: ["Library"] });
    if (path === "/faqs/Library") return success({ faqs: [{ Question: "Library hours?" }] });
    if (path === "/start_conversation") return success({ conversation_id: "id", conversation_token: "token" });
    assert.equal(JSON.parse(options.body).message, "Library hours?");
    return success({ response: "Library answer", suggestions: [] });
  });
  await state.run('openExplorer("faqs")');
  const panel = state.elements.get("#explore-content");
  assert.equal(state.elements.get("#explore-dialog").open, true);
  assert.equal(panel.children[0].children[0].textContent, "Library");
  await panel.children[0].listeners.click();
  assert.equal(panel.children[0].children[0].textContent, "Library hours?");
  await panel.children[0].listeners.click();
  assert.equal(state.elements.get("#explore-dialog").open, false);
  assert.equal(state.elements.get("#welcome").hidden, true);
  assert.equal(state.elements.get("#chatbox").hidden, false);
  assert.equal(state.elements.get("#chatbox").children.at(-1).textContent, "Library answer");
  assert.deepEqual(paths, ["/faq_classes", "/faqs/Library", "/start_conversation", "/ask/id"]);
});

test("location search encodes input and keeps result names literal", async () => {
  const calls = [];
  const location = "<img src=x onerror=alert(1)>";
  const state = harness(async path => {
    calls.push(path);
    return success({ navigation: [{ Location: location, Building: "Main building", Level: "Level 1" }] });
  });
  state.run('openExplorer("locations")');
  await state.run('searchLocations("Blood Collection Room 2 (Special)")');
  assert.equal(calls[0], "/navigation/Blood%20Collection%20Room%202%20(Special)");
  const row = state.elements.get("#explore-content").children[0];
  assert.equal(row.children[0].textContent, location);
  assert.equal(row.children[1].textContent, "Main building / Level 1");
});

test("a late FAQ response cannot overwrite a different explorer view", async () => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const state = harness(async () => { await pending; return success({ classes: ["Late category"] }); });
  const first = state.run('openExplorer("faqs")');
  state.run('openExplorer("locations")');
  release();
  await first;
  assert.equal(state.elements.get("#explore-title").textContent, "Find your next stop.");
  assert.equal(state.elements.get("#explore-content").children[1].children[0].textContent, "Pharmacy");
  assert.equal(state.elements.get("#explore-content").attributes["aria-busy"], "false");
});

test("closing the explorer ignores pending results and clears busy state", async () => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const state = harness(async () => { await pending; return success({ classes: ["Late category"] }); });
  const first = state.run('openExplorer("faqs")');
  state.run("explorer.close()");
  release();
  await first;
  assert.equal(state.elements.get("#explore-dialog").open, false);
  assert.equal(state.elements.get("#explore-content").attributes["aria-busy"], "false");
});

test("no matching location offers a useful recovery message", async () => {
  const state = harness(async () => ({ ok: false, status: 404, json: async () => ({ message: "Location not found." }) }));
  state.run('openExplorer("locations")');
  await state.run('searchLocations("missing place")');
  assert.match(state.elements.get("#explore-content").children[0].textContent, /Try a shorter name/);
});

test("start fresh deletes the server conversation and restores the welcome", async () => {
  const calls = [];
  const state = harness(async (path, options) => {
    calls.push({ path, options });
    if (path === "/start_conversation") return success({ conversation_id: "id", conversation_token: "token" });
    if (options.method === "DELETE") return { ok: true, status: 204 };
    return success({ response: "Answer", suggestions: [] });
  });
  state.elements.get("#user-input").value = "Question";
  await state.run("sendMessage()");
  await state.run("resetConversation()");
  assert.equal(calls.at(-1).path, "/history/id");
  assert.equal(calls.at(-1).options.method, "DELETE");
  assert.equal(calls.at(-1).options.headers.Authorization, "Bearer token");
  assert.equal(state.run("conversation"), null);
  assert.equal(state.elements.get("#chatbox").children.length, 0);
  assert.equal(state.elements.get("#chatbox").hidden, true);
  assert.equal(state.elements.get("#welcome").hidden, false);
  assert.equal(state.elements.get("#character-count").textContent, "0 / 2,000");
  assert.equal(state.elements.get("#typing").hidden, true);
});

test("failed deletion keeps the existing conversation and its messages", async () => {
  const state = harness(async (path, options) => {
    if (path === "/start_conversation") return success({ conversation_id: "id", conversation_token: "token" });
    if (options.method === "DELETE") throw new Error("Network failed");
    return success({ response: "Answer", suggestions: [] });
  });
  state.elements.get("#user-input").value = "Question";
  await state.run("sendMessage()");
  await state.run("resetConversation()");
  assert.equal(state.run("conversation.conversation_id"), "id");
  assert.equal(state.elements.get("#chatbox").children.length, 2);
  assert.match(state.elements.get("#status").textContent, /Could not clear/);
  assert.equal(state.elements.get("#reset-btn").disabled, false);
});

test("theme controls update their accessible name in both directions", () => {
  const state = harness();
  const button = state.elements.get("#toggle-dark");
  button.listeners.click({ currentTarget: button });
  assert.equal(button.attributes["aria-label"], "Use light theme");
  assert.equal(button.attributes["aria-pressed"], "true");
  button.listeners.click({ currentTarget: button });
  assert.equal(button.attributes["aria-label"], "Use dark theme");
  assert.equal(button.attributes["aria-pressed"], "false");
});
