// Streams Assistant replies. The form works without this file (it posts to
// /assistant/send and waits); with it, the same form posts to /assistant/stream
// and the reply appears as it's generated.
//
// Server-Sent Events arrive over a fetch() POST rather than EventSource, which
// can only GET: the message and CSRF token travel in the request body and
// header, never in a URL.
//
// Everything from the server is inserted with textContent, never innerHTML, so
// nothing the model writes can become markup or script on this page.
"use strict";

(function () {
  const TITLE_LENGTH = 60; // conversations.TITLE_LENGTH
  const ANSWER_BREAK = "\n\n"; // pages/assistant.py ANSWER_BREAK
  // A message typed but not sent because the session had expired: kept for
  // this tab only, put back in the box after signing in, then forgotten.
  const UNSENT_KEY = "agrios-unsent-message";

  function remember(message) {
    try {
      sessionStorage.setItem(UNSENT_KEY, message);
    } catch (err) {
      // Storage can be unavailable (private mode, blocked); the message is lost as before.
    }
  }

  function restoreUnsent() {
    const textarea = document.querySelector("[data-assistant-form] textarea[name=message]");
    if (!textarea) return;
    try {
      const saved = sessionStorage.getItem(UNSENT_KEY);
      if (saved && !textarea.value) textarea.value = saved;
      sessionStorage.removeItem(UNSENT_KEY);
    } catch (err) {
      // Nothing to restore.
    }
  }

  function parseEvents(buffer, onEvent) {
    // Events are separated by a blank line; the remainder is an incomplete one.
    const blocks = buffer.split(/\r?\n\r?\n/);
    const rest = blocks.pop();
    for (const block of blocks) {
      let event = "message";
      const data = [];
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith(":")) continue; // keep-alive comment
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "event") event = value;
        else if (field === "data") data.push(value);
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
    return rest;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function titleFor(message) {
    const text = message.split(/\s+/).filter(Boolean).join(" ");
    return text.length <= TITLE_LENGTH ? text : text.slice(0, TITLE_LENGTH - 1).trimEnd() + "…";
  }

  function addToThreadList(url, message) {
    const nav = document.querySelector(".threads");
    if (!nav) return;
    let list = nav.querySelector(".list");
    if (!list) {
      list = el("ul", "list");
      nav.appendChild(list);
    }
    const item = el("li");
    const link = el("a", "", titleFor(message));
    link.href = url;
    link.setAttribute("aria-current", "page");
    item.appendChild(link);
    list.prepend(item);
    const fresh = nav.querySelector(".new-thread");
    if (fresh) fresh.removeAttribute("aria-current");
  }

  async function send(form) {
    const textarea = form.querySelector("textarea[name=message]");
    const button = form.querySelector("button[type=submit]");
    const status = form.querySelector(".status");
    const log = document.getElementById("conversation");
    const errorBox = document.getElementById("chat-error");
    const template = document.getElementById("step-template");
    const message = textarea.value.trim();
    // requestSubmit() submits even with the button disabled, so Enter needs its
    // own guard against a second message while one is answering.
    if (!message || form.dataset.sending) return;
    form.dataset.sending = "true";

    const body = new FormData(form);
    const isNew = !body.get("thread_id");
    const turn = el("div", "turn");
    const you = el("p", "you");
    you.append(el("span", "visually-hidden", "You:"), message);
    const steps = el("ul", "steps");
    steps.setAttribute("aria-label", "What the assistant checked");
    const answer = el("div", "answer");
    const answerText = document.createTextNode("");
    answer.append(el("span", "visually-hidden", "Agrios:"), answerText);
    turn.append(you, steps, answer);
    log.appendChild(turn);
    turn.scrollIntoView({ block: "end" });

    errorBox.hidden = true;
    button.disabled = true;
    status.textContent = "Thinking…";
    log.setAttribute("aria-busy", "true");
    let failed = false;
    // Text the model writes before a tool call is kept, and what follows the
    // call starts a new paragraph, the same as the saved conversation shows it.
    let afterTool = false;

    const handlers = {
      thread(data) {
        form.querySelector("input[name=thread_id]").value = data.id;
        if (isNew) {
          history.replaceState(null, "", data.url);
          addToThreadList(data.url, message);
        }
      },
      token(text) {
        if (afterTool && answerText.data) answerText.data += ANSWER_BREAK;
        afterTool = false;
        answerText.data += text;
        status.textContent = "Answering…";
      },
      tool(data) {
        afterTool = true;
        const item = template.content.firstElementChild.cloneNode(true);
        item.dataset.id = data.id;
        item.querySelector(".label").textContent = data.label;
        steps.appendChild(item);
        status.textContent = data.label + "…";
      },
      tool_done(data) {
        const item = [...steps.children].find((li) => li.dataset.id === data.id);
        if (!item) return;
        item.classList.remove("running");
        item.classList.add(data.error ? "failed" : "ok");
        item.querySelector(".icon").textContent = data.error ? "✖" : "✔";
        if (data.error) item.querySelector(".step-detail").textContent = "failed: " + data.error;
      },
      error(data) {
        failed = true;
        errorBox.querySelector(".text").textContent = data.message;
        errorBox.hidden = false;
      },
      done() {},
    };

    try {
      const response = await fetch(form.dataset.stream, {
        method: "POST",
        body,
        headers: { "x-csrf-token": body.get("csrf_token") },
        credentials: "same-origin",
      });
      const type = response.headers.get("content-type") || "";
      if (!response.ok || !type.startsWith("text/event-stream")) {
        // Signed out meanwhile: the guard redirected to the sign-in page.
        if (response.redirected) {
          remember(message);
          window.location.assign(response.url);
          return;
        }
        throw new Error("The console answered " + response.status + ".");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer = parseEvents(buffer + decoder.decode(value, { stream: true }), (event, data) => {
          if (handlers[event]) handlers[event](data);
        });
      }
    } catch (err) {
      handlers.error({ message: "The reply was cut off: " + err.message + " Reload the page to see what was saved." });
    } finally {
      if (!answerText.data && !steps.children.length) turn.classList.add("unanswered");
      if (!answerText.data) answer.remove();
      if (!steps.children.length) steps.remove();
      if (!failed) textarea.value = "";
      delete form.dataset.sending;
      button.disabled = false;
      status.textContent = "";
      log.setAttribute("aria-busy", "false");
      textarea.focus();
    }
  }

  // On a full load and after each htmx swap (sign-in is a boosted form).
  document.addEventListener("DOMContentLoaded", restoreUnsent);
  document.addEventListener("htmx:afterSettle", restoreUnsent);

  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement) || !form.matches("[data-assistant-form]")) return;
    event.preventDefault();
    send(form);
  });

  // Enter sends, Shift+Enter starts a new line: the usual chat convention.
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
    if (!(target instanceof HTMLTextAreaElement) || !target.closest("[data-assistant-form]")) return;
    event.preventDefault();
    target.form.requestSubmit();
  });
})();
