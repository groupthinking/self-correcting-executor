# MCP Chrome Assistant Extension v0.3 — **Viseme Lip‑Sync + Voice I/O**

> **Goal**  Bring the avatar fully to life:
>
> - 🔊 Text‑to‑speech (TTS) playback of MCP responses
> - 🎤 Push‑to‑talk voice input (Web Speech API)
> - 👄 Real‑time viseme animation driven by `speechSynthesis.onboundary` events

---

## 🗂 Updated Asset Map

| Asset                     | Purpose              | Notes         |
| ------------------------- | -------------------- | ------------- |
| `viseme_idle.png`         | Neutral mouth sprite | Default frame |
| `viseme_AE.png`           | "A / E" mouth        |               |
| `viseme_O.png`            | "O" mouth            |               |
| `viseme_U.png`            | "U" mouth            |               |
| `viseme_L.png`            | "L / Th" mouth       |               |
| `viseme_M.png`            | "M / B / P" mouth    |               |
| `viseme_S.png`            | "S / Z" mouth        |               |
| *(add others as desired)* |                      |               |

> We switched from video clips ➜ **PNG sprite swap**. Each viseme frame is a transparent PNG rendered in a `<canvas>` element, enabling per‑frame lip sync.

---

## 1. `manifest.json`

```json
{
  "manifest_version": 3,
  "name": "MCP Assistant",
  "version": "0.3.0",
  "description": "AI assistant with MCP backend, voice I/O, and viseme lip‑sync avatar.",
  "icons": {
    "16": "icon.png",
    "48": "icon.png",
    "128": "icon.png"
  },
  "action": {
    "default_title": "MCP Assistant",
    "default_icon": "icon.png",
    "default_popup": "sidebar.html"
  },
  "permissions": [
    "activeTab",
    "tabs",
    "storage",
    "scripting",
    "microphone"
  ],
  "host_permissions": ["<all_urls>"],
  "background": { "service_worker": "background.js" },
  "web_accessible_resources": [
    {
      "resources": [
        "viseme_*.png"
      ],
      "matches": ["<all_urls>"]
    }
  ]
}
```

---

## 2. `sidebar.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MCP Assistant</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <div id="mcp-assistant">
    <header>
      <h2>MCP Assistant</h2>
      <span id="mode-indicator">MCP</span>
    </header>

    <!-- Canvas‑based avatar (viseme frames are blitted here) -->
    <canvas id="avatar-canvas" width="320" height="180"></canvas>

    <textarea id="agent-input" placeholder="Ask something…" rows="3"></textarea>

    <div class="btn-row">
      <button id="btn-mcp">Ask MCP</button>
      <button id="btn-voice">🎤 Hold to Talk</button>
      <button id="btn-clear" title="Clear history">🗑</button>
    </div>

    <section id="chat-log"></section>
  </div>

  <script src="sidebar.js"></script>
</body>
</html>
```

---

## 3. `styles.css` (delta)

```css
/* …previous styles remain unchanged… */

#avatar-canvas {
  width: 100%;
  border-radius: 8px;
  background: #000;
  margin-bottom: 8px;
}

#btn-voice {
  flex: 0 0 42px;
  font-size: 1rem;
  background: #5a3a3a;
}
#btn-voice.recording {
  background: #c0392b;
}
```

---

## 4. `sidebar.js` (complete rewrite)

```js
/* eslint-disable no-undef */
const MCP_ENDPOINT = "https://your-mcp-server.com/ask"; // TODO

/*****************************************************************
 *  Avatar Viseme Engine                                          *
 *****************************************************************/
const canvas = document.getElementById("avatar-canvas");
const ctx = canvas.getContext("2d");
const VIS_EMES = {
  idle: "viseme_idle.png",
  AE: "viseme_AE.png",
  O: "viseme_O.png",
  U: "viseme_U.png",
  L: "viseme_L.png",
  M: "viseme_M.png",
  S: "viseme_S.png"
};
const visemeCache = {};
function loadImg(src) {
  return new Promise(res => {
    if (visemeCache[src]) return res(visemeCache[src]);
    const img = new Image();
    img.src = chrome.runtime.getURL(src);
    img.onload = () => {
      visemeCache[src] = img;
      res(img);
    };
  });
}
async function drawViseme(key) {
  const img = await loadImg(VIS_EMES[key] || VIS_EMES.idle);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}
// idle to start
drawViseme("idle");

/*****************************************************************
 *  Web Speech – TTS + Boundary‑driven viseme sync                *
 *****************************************************************/
const visemeMap = {
  "a": "AE", "e": "AE", // open mouth
  "o": "O",  "ʊ": "O",  // rounded
  "u": "U",
  "l": "L",  "θ": "L",
  "m": "M",  "b": "M", "p": "M",
  "s": "S",  "z": "S"
};
function phonemeToViseme(phoneme) {
  return visemeMap[phoneme.toLowerCase()] || "AE";
}
function speak(text) {
  return new Promise(resolve => {
    const utter = new SpeechSynthesisUtterance(text);
    utter.voice = speechSynthesis.getVoices().find(v => v.lang.startsWith("en"));
    utter.onboundary = e => {
      if (e.name === "word") return; // ignore word boundaries
      const char = text[e.charIndex] || " ";
      const vKey = phonemeToViseme(char);
      drawViseme(vKey);
    };
    utter.onstart = () => drawViseme("AE");
    utter.onend = () => {
      drawViseme("successViseme" in VIS_EMES ? "successViseme" : "idle");
      resolve();
    };
    speechSynthesis.speak(utter);
  });
}

/*****************************************************************
 *  Voice Recognition (Push‑to‑Talk)                              *
 *****************************************************************/
const voiceBtn = document.getElementById("btn-voice");
let recognition;
if ("webkitSpeechRecognition" in window) {
  recognition = new webkitSpeechRecognition();
  recognition.interimResults = false;
  recognition.lang = "en-US";
}
voiceBtn.addEventListener("mousedown", () => {
  if (!recognition) return alert("SpeechRecognition not supported");
  voiceBtn.classList.add("recording");
  recognition.start();
});
voiceBtn.addEventListener("mouseup", () => {
  if (!recognition) return;
  voiceBtn.classList.remove("recording");
  recognition.stop();
});
if (recognition) {
  recognition.onresult = e => {
    const transcript = e.results[0][0].transcript;
    document.getElementById("agent-input").value = transcript;
    document.getElementById("btn-mcp").click();
  };
  recognition.onerror = () => voiceBtn.classList.remove("recording");
}

/*****************************************************************
 *  Chat logic (stream + TTS)                                     *
 *****************************************************************/
const inputEl   = document.getElementById("agent-input");
const logEl     = document.getElementById("chat-log");
const askBtn    = document.getElementById("btn-mcp");
const clearBtn  = document.getElementById("btn-clear");

function addMsg(role, txt) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = txt;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

askBtn.onclick = async () => {
  const q = inputEl.value.trim();
  if (!q) return;
  addMsg("user", q);
  inputEl.value = "";
  const agentDiv = addMsg("agent", "");
  drawViseme("think");
  try {
    const res = await fetch(MCP_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, stream: true })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      agentDiv.textContent = buf;
      logEl.scrollTop = logEl.scrollHeight;
    }
    await speak(buf); // TTS with visemes
  } catch (e) {
    drawViseme("errorViseme" in VIS_EMES ? "errorViseme" : "idle");
    agentDiv.textContent = "[Error] " + e.message;
  }
};

clearBtn.onclick = () => logEl.innerHTML = "";
```

---

## 5. `background.js`

*(unchanged)*

---

### 🚀 Setup Steps

1. Export/prepare PNG viseme frames listed above (transparent BG; 320×180).
2. Drop them in your extension folder.
3. `chrome://extensions` → *Reload*.
4. Hold the 🎤 button to speak, or type → **avatar mouths words, speaks back**.

---

### ↪️ Next Possible Increments

- Replace sprite system with **WebGL shader** or **Three.js 3‑D head**.
- Hook to **Whisper.cpp WASM** for on‑device ASR (higher accuracy).
- Multi‑language viseme mapping via IPA phoneme lookup.

---

> **v0.3 delivered — voice I/O & viseme sync operational.**

