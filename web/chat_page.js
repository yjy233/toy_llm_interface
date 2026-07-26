const defaults = {
  model: "qwen3.5-0.8b",
  device: "cpu",
  maxNewTokens: 16
};

const modelInput = document.getElementById("model");
const deviceInput = document.getElementById("device");
const tokensInput = document.getElementById("tokens");
const temperatureInput = document.getElementById("temperature");
const streamInput = document.getElementById("stream");
const thinkingInput = document.getElementById("thinking");
const messagesEl = document.getElementById("messages");
const emptyEl = document.getElementById("empty");
const form = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const imageInput = document.getElementById("image-input");
const attachButton = document.getElementById("attach");
const attachmentsEl = document.getElementById("attachments");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
const statusEl = document.getElementById("status");

const messages = [];
const selectedImages = [];
let controller = null;
let imagesEnabled = false;

attachButton.disabled = true;

function normalizeDevice(device) {
  return device === "mps" ? "mps:0" : device;
}

function hasDeviceOption(device) {
  return Array.from(deviceInput.options).some((option) => option.value === device);
}

function applyConfig(config) {
  const device = normalizeDevice(config.device || defaults.device);
  modelInput.value = config.model || defaults.model;
  tokensInput.value = String(config.max_new_tokens || defaults.maxNewTokens);
  imagesEnabled = Boolean(config.has_mmproj);
  attachButton.disabled = !imagesEnabled;
  attachButton.title = imagesEnabled ? "Attach images" : "Start the gateway with --mmproj to attach images";
  if (hasDeviceOption(device)) {
    deviceInput.value = device;
  }
  streamInput.disabled = false;
}

async function loadConfig() {
  applyConfig(defaults);
  try {
    const response = await fetch("/chat_page/config", {cache: "no-store"});
    if (!response.ok) {
      return;
    }
    applyConfig(await response.json());
  } catch (_) {
  }
}

function setStatus(value) {
  statusEl.textContent = value;
}

function setBusy(value) {
  sendButton.disabled = value;
  stopButton.disabled = !value;
  promptInput.disabled = value;
  imageInput.disabled = value || !imagesEnabled;
  attachButton.disabled = value || !imagesEnabled;
  setStatus(value ? "Busy" : "Ready");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setBubbleText(bubble, text) {
  let textNode = bubble.querySelector(".message-text");
  if (!textNode) {
    textNode = document.createElement("div");
    textNode.className = "message-text";
    bubble.prepend(textNode);
  }
  textNode.textContent = text;
}

function addBubble(role, text, images = []) {
  emptyEl.hidden = true;
  const node = document.createElement("div");
  node.className = "message " + role;
  const textNode = document.createElement("div");
  textNode.className = "message-text";
  textNode.textContent = text;
  node.appendChild(textNode);
  if (images.length > 0) {
    const media = document.createElement("div");
    media.className = "message-images";
    for (const image of images) {
      const imageNode = document.createElement("img");
      imageNode.src = image.dataUrl;
      imageNode.alt = image.name || "attached image";
      media.appendChild(imageNode);
    }
    node.appendChild(media);
  }
  messagesEl.appendChild(node);
  scrollToBottom();
  return node;
}

function renderAttachments() {
  attachmentsEl.replaceChildren();
  attachmentsEl.hidden = selectedImages.length === 0;
  selectedImages.forEach((image, index) => {
    const item = document.createElement("div");
    item.className = "attachment";

    const preview = document.createElement("img");
    preview.src = image.dataUrl;
    preview.alt = image.name || "attached image";
    item.appendChild(preview);

    const name = document.createElement("span");
    name.textContent = image.name || "image";
    item.appendChild(name);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "attachment-remove";
    remove.setAttribute("aria-label", "Remove image");
    remove.textContent = "x";
    remove.addEventListener("click", () => {
      selectedImages.splice(index, 1);
      renderAttachments();
      promptInput.focus();
    });
    item.appendChild(remove);
    attachmentsEl.appendChild(item);
  });
}

function clearAttachments() {
  selectedImages.splice(0, selectedImages.length);
  imageInput.value = "";
  renderAttachments();
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("failed to read image"));
    reader.readAsDataURL(file);
  });
}

function userContent(text, images) {
  if (images.length === 0) {
    return text;
  }
  const parts = [];
  if (text) {
    parts.push({type: "text", text});
  }
  for (const image of images) {
    parts.push({
      type: "image_url",
      image_url: {
        url: image.dataUrl,
        detail: "auto"
      }
    });
  }
  return parts;
}

function requestBody(history) {
  const maxTokens = Math.max(1, Number.parseInt(tokensInput.value || "1", 10));
  const temperature = Number.parseFloat(temperatureInput.value);
  const body = {
    model: modelInput.value.trim() || defaults.model,
    messages: history,
    max_completion_tokens: maxTokens,
    stream: streamInput.checked && !streamInput.disabled,
    enable_thinking: thinkingInput.checked,
    device: deviceInput.value
  };
  if (Number.isFinite(temperature)) {
    body.temperature = temperature;
  }
  return body;
}

function contentFromPayload(payload) {
  return payload && payload.choices && payload.choices[0] &&
    payload.choices[0].message && payload.choices[0].message.content || "";
}

async function readStream(response, bubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }
    buffer += decoder.decode(chunk.value, {stream: true});
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const eventText = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const rawLine of eventText.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) {
          continue;
        }
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          return text;
        }
        const payload = JSON.parse(data);
        const delta = payload && payload.choices && payload.choices[0] &&
          payload.choices[0].delta && payload.choices[0].delta.content || "";
        if (delta) {
          text += delta;
          setBubbleText(bubble, text);
          scrollToBottom();
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }
  return text;
}

async function sendMessage(text, images) {
  const content = userContent(text, images);
  const nextMessages = messages.concat([{role: "user", content}]);
  addBubble("user", text, images);
  const assistantBubble = addBubble("assistant", "");
  controller = new AbortController();
  setBusy(true);
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(requestBody(nextMessages)),
      signal: controller.signal
    });
    if (!response.ok) {
      let errorText = "HTTP " + response.status;
      try {
        const payload = await response.json();
        errorText = payload.error && payload.error.message || errorText;
      } catch (_) {
      }
      throw new Error(errorText);
    }
    let answer = "";
    if (streamInput.checked) {
      answer = await readStream(response, assistantBubble);
    } else {
      const payload = await response.json();
      answer = contentFromPayload(payload);
      setBubbleText(assistantBubble, answer);
    }
    messages.push({role: "user", content});
    messages.push({role: "assistant", content: answer});
    scrollToBottom();
  } catch (error) {
    assistantBubble.remove();
    addBubble("error", error.name === "AbortError" ? "Stopped" : error.message);
  } finally {
    controller = null;
    setBusy(false);
    promptInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  const images = selectedImages.slice();
  if ((!text && images.length === 0) || controller) {
    return;
  }
  promptInput.value = "";
  clearAttachments();
  void sendMessage(text, images);
});

attachButton.addEventListener("click", () => {
  if (!imagesEnabled || controller) {
    return;
  }
  imageInput.click();
});

imageInput.addEventListener("change", async () => {
  const files = Array.from(imageInput.files || []);
  for (const file of files) {
    if (!file.type.startsWith("image/")) {
      continue;
    }
    if (file.size > 8 * 1024 * 1024) {
      setStatus("Image too large");
      continue;
    }
    const dataUrl = await readFileAsDataUrl(file);
    selectedImages.push({
      name: file.name,
      type: file.type,
      dataUrl
    });
  }
  imageInput.value = "";
  renderAttachments();
  promptInput.focus();
});

stopButton.addEventListener("click", () => {
  if (controller) {
    controller.abort();
  }
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

void loadConfig();
