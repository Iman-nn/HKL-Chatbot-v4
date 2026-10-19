"use strict";

const chatbox = document.querySelector("#chatbox");
const input = document.querySelector("#user-input");
const send = document.querySelector("#send-btn");
const status = document.querySelector("#status");
const reset = document.querySelector("#reset-btn");
const clearButton = document.querySelector("#clear-chat");
const welcome = document.querySelector("#welcome");
const conversationArea = document.querySelector("#conversation-area");
const typing = document.querySelector("#typing");
const characterCount = document.querySelector("#character-count");
const explorer = document.querySelector("#explore-dialog");
const explorerContent = document.querySelector("#explore-content");
const explorerTitle = document.querySelector("#explore-title");
const explorerSubtitle = document.querySelector("#explore-subtitle");
const explorerBack = document.querySelector("#explore-back");
const locationForm = document.querySelector("#location-form");
const locationQuery = document.querySelector("#location-query");
const privacy = document.querySelector("#privacy-dialog");

let conversation = null;
let busy = false;
let recognition = null;
let explorerRequest = 0;

class ApiError extends Error {
  constructor(message, code) {
    super(message);
    this.code = code;
  }
}

async function api(path, { method = "GET", body, authenticated = false } = {}) {
  const headers = { Accept: "application/json" };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (authenticated) headers.Authorization = "Bearer " + conversation.conversation_token;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 35000);
  try {
    const response = await fetch(path, {
      method, headers, signal: controller.signal, credentials: "omit",
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: "no-store",
    });
    if (response.status === 204) return null;
    const data = await response.json();
    if (!response.ok) throw new ApiError(data.error || data.message || "Request failed. Please try again.", response.status);
    return data;
  } finally {
    clearTimeout(timeout);
  }
}

function updateCharacterCount() {
  characterCount.textContent = input.value.length.toLocaleString("en-US") + " / 2,000";
}

function scrollConversation() {
  conversationArea.scrollTop = conversationArea.scrollHeight;
}

function appendMessage(text, author = "bot") {
  welcome.hidden = true;
  chatbox.hidden = false;
  const message = document.createElement("div");
  message.className = "message " + author;
  message.setAttribute("role", "article");
  message.setAttribute("aria-label", author === "user" ? "You" : "HKL Guide");
  // User, CSV and provider content stays literal text, including in explorer results.
  message.textContent = text;
  chatbox.append(message);
  while (chatbox.children.length > 150) chatbox.firstElementChild.remove();
  scrollConversation();
}

function setBusy(value) {
  busy = value;
  send.disabled = value;
  reset.disabled = value;
  clearButton.disabled = value;
  typing.hidden = !value;
  chatbox.setAttribute("aria-busy", String(value));
}

function showSuggestions(suggestions) {
  const container = document.querySelector("#suggestions");
  container.replaceChildren();
  for (const suggestion of suggestions.slice(0, 4)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion";
    button.textContent = suggestion;
    button.addEventListener("click", () => {
      input.value = suggestion;
      updateCharacterCount();
      input.focus();
    });
    container.append(button);
  }
}

async function sendMessage(event) {
  event?.preventDefault();
  const message = input.value.trim();
  if (busy) return;
  if (!message || message.length > 2000) {
    status.textContent = "Enter a message of up to 2,000 characters.";
    return;
  }
  setBusy(true);
  status.textContent = "Finding an answer...";
  let submitted = false;
  try {
    // Initialization completes before asking; the busy guard prevents duplicate sends.
    if (!conversation) conversation = await api("/start_conversation", { method: "POST", body: {} });
    appendMessage(message, "user");
    submitted = true;
    input.value = "";
    updateCharacterCount();
    showSuggestions([]);
    const result = await api("/ask/" + encodeURIComponent(conversation.conversation_id), {
      method: "POST", body: { message }, authenticated: true,
    });
    appendMessage(result.source === "external_ai"
      ? "AI-generated answer. Please verify with hospital staff.\n" + result.response
      : result.response);
    showSuggestions(result.suggestions || []);
    status.textContent = "";
  } catch (error) {
    if (error.code === 404) {
      conversation = null;
      status.textContent = "This conversation has expired. Send your question again to start a new one.";
    } else {
      status.textContent = error instanceof ApiError ? error.message : "Connection failed. Please try again.";
    }
    // Preserve the question without silently retrying a potentially billable request.
    if (submitted && !input.value) input.value = message;
    updateCharacterCount();
  } finally {
    setBusy(false);
    input.focus();
    scrollConversation();
  }
}

async function resetConversation() {
  if (busy) return;
  setBusy(true);
  try {
    if (conversation) {
      try {
        await api("/history/" + encodeURIComponent(conversation.conversation_id), {
          method: "DELETE", authenticated: true,
        });
      } catch (error) {
        if (error.code !== 404) throw error;
      }
    }
    conversation = null;
    chatbox.replaceChildren();
    chatbox.hidden = true;
    welcome.hidden = false;
    showSuggestions([]);
    input.value = "";
    updateCharacterCount();
    status.textContent = "A fresh start. What would you like to know?";
    conversationArea.scrollTop = 0;
    input.focus();
  } catch {
    status.textContent = "Could not clear the conversation. Please try again.";
  } finally {
    setBusy(false);
  }
}

function panelMessage(text) {
  const message = document.createElement("p");
  message.className = "panel-message";
  message.textContent = text;
  explorerContent.replaceChildren();
  explorerContent.append(message);
}

function explorerItem(title, description, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "explore-item";
  const heading = document.createElement("strong");
  heading.textContent = title;
  button.append(heading);
  if (description) {
    const detail = document.createElement("small");
    detail.textContent = description;
    button.append(detail);
  }
  button.addEventListener("click", onClick);
  return button;
}

async function loadExplorer(path, render) {
  const requestId = ++explorerRequest;
  panelMessage("Getting that information for you...");
  explorerContent.setAttribute("aria-busy", "true");
  try {
    const data = await api(path);
    // Ignore late responses after closing, switching view, or starting another search.
    if (requestId !== explorerRequest || !explorer.open) return;
    explorerContent.replaceChildren();
    render(data);
  } catch (error) {
    if (requestId !== explorerRequest || !explorer.open) return;
    panelMessage(error.code === 404 && path.startsWith("/navigation/")
      ? "No matching locations. Try a shorter name, such as Pharmacy or Radiology."
      : error instanceof ApiError ? error.message : "We could not load this information. Please try again.");
  } finally {
    if (requestId === explorerRequest) explorerContent.setAttribute("aria-busy", "false");
  }
}

function chooseQuestion(question) {
  input.value = question;
  updateCharacterCount();
  explorer.close();
  input.focus();
  // Browsing remains available during a response. A selected question waits in
  // the composer rather than replacing an in-flight request.
  if (!busy) return sendMessage();
}

function loadCategory(category) {
  explorerTitle.textContent = category;
  explorerSubtitle.textContent = "Choose a question to open its answer in your conversation.";
  explorerBack.hidden = false;
  return loadExplorer("/faqs/" + encodeURIComponent(category), data => {
    if (!data.faqs.length) return panelMessage("There are no questions in this category yet.");
    for (const faq of data.faqs) {
      explorerContent.append(explorerItem(faq.Question, "Ask HKL Guide", () => chooseQuestion(faq.Question)));
    }
  });
}

function loadCategories() {
  explorerTitle.textContent = "A question? Start here.";
  explorerSubtitle.textContent = "Explore hospital information by topic, then choose a question.";
  explorerBack.hidden = true;
  return loadExplorer("/faq_classes", data => {
    if (!data.classes.length) return panelMessage("There are no FAQ categories available right now.");
    for (const category of data.classes) {
      explorerContent.append(explorerItem(category, "Explore questions in this topic", () => loadCategory(category)));
    }
  });
}

function searchLocations(query) {
  query = query.trim();
  if (!query || query.length > 200) {
    panelMessage("Enter a location name of up to 200 characters.");
    return;
  }
  locationQuery.value = query;
  return loadExplorer("/navigation/" + encodeURIComponent(query), data => {
    if (!data.navigation.length) return panelMessage("No matching locations. Try another name.");
    for (const location of data.navigation) {
      const detail = [location.Building, location.Level].filter(Boolean).join(" / ");
      explorerContent.append(explorerItem(location.Location, detail, () => chooseQuestion(location.Location)));
    }
  });
}

function openExplorer(mode) {
  // Invalidating first also handles switching modes while an earlier request is pending.
  explorerRequest++;
  explorerContent.setAttribute("aria-busy", "false");
  locationForm.hidden = mode !== "locations";
  explorerBack.hidden = true;
  if (mode === "faqs") {
    const pending = loadCategories();
    if (!explorer.open) explorer.showModal();
    return pending;
  }
  explorerTitle.textContent = "Find your next stop.";
  explorerSubtitle.textContent = "Search for a clinic or facility. Choose a result for directions in your conversation.";
  locationQuery.value = "";
  panelMessage("Not sure where to start? Try one of these places.");
  for (const location of ["Pharmacy", "Radiology", "Blood Collection"]) {
    explorerContent.append(explorerItem(location, "Search hospital locations", () => searchLocations(location)));
  }
  if (!explorer.open) explorer.showModal();
  locationQuery.focus();
}

document.querySelector("#chat-form").addEventListener("submit", sendMessage);
input.addEventListener("input", updateCharacterCount);
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) sendMessage(event);
});
reset.addEventListener("click", resetConversation);
clearButton.addEventListener("click", resetConversation);
document.querySelector("#chat-nav").addEventListener("click", () => input.focus());

for (const selector of ["#browse-faqs", "#quick-faqs"]) {
  document.querySelector(selector).addEventListener("click", () => openExplorer("faqs"));
}
for (const selector of ["#browse-locations", "#quick-directions"]) {
  document.querySelector(selector).addEventListener("click", () => openExplorer("locations"));
}
for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => {
    if (busy) return;
    input.value = button.getAttribute("data-question");
    updateCharacterCount();
    sendMessage();
  });
}
locationForm.addEventListener("submit", event => {
  event.preventDefault();
  searchLocations(locationQuery.value);
});
explorerBack.addEventListener("click", loadCategories);
document.querySelector("#close-explore").addEventListener("click", () => explorer.close());
explorer.addEventListener("close", () => {
  explorerRequest++;
  explorerContent.setAttribute("aria-busy", "false");
});
for (const selector of ["#privacy-btn", "#privacy-inline"]) {
  document.querySelector(selector).addEventListener("click", () => privacy.showModal());
}
document.querySelector("#close-privacy").addEventListener("click", () => privacy.close());

document.querySelector("#toggle-dark").addEventListener("click", event => {
  const dark = document.body.classList.toggle("dark-mode");
  event.currentTarget.setAttribute("aria-pressed", String(dark));
  event.currentTarget.setAttribute("aria-label", dark ? "Use light theme" : "Use dark theme");
});

const voiceButton = document.querySelector("#voice-btn");
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "en-MY";
  recognition.continuous = false;
  recognition.interimResults = false;
  let recognizing = false;
  const endRecognition = () => {
    recognizing = false;
    voiceButton.setAttribute("aria-pressed", "false");
    voiceButton.setAttribute("aria-label", "Start voice input");
  };
  recognition.onresult = event => {
    input.value = event.results[0][0].transcript.slice(0, 2000);
    updateCharacterCount();
    input.focus();
  };
  recognition.onend = endRecognition;
  recognition.onerror = () => {
    endRecognition();
    status.textContent = "Voice input is unavailable. Please type your question.";
  };
  voiceButton.addEventListener("click", () => {
    try {
      if (recognizing) recognition.stop();
      else {
        recognition.start();
        recognizing = true;
        voiceButton.setAttribute("aria-pressed", "true");
        voiceButton.setAttribute("aria-label", "Stop voice input");
      }
    } catch {
      endRecognition();
    }
  });
} else {
  voiceButton.disabled = true;
  voiceButton.title = "Voice input is unavailable in this browser";
}
updateCharacterCount();
