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
  };
  const context = vm.createContext({ document, window: {}, fetch, AbortController, setTimeout, clearTimeout });
  vm.runInContext(fs.readFileSync("static/chat.js", "utf8"), context);
  return { context, elements, run: code => vm.runInContext(code, context) };
}

const success = data => ({ ok: true, status: 200, json: async () => data });

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
