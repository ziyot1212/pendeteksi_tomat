
import os

html_path = os.path.join('templates', 'index.html')

with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Hapus widget lama jika sudah terpasang
if '<!-- START_TOMATO_AI_CHAT -->' in content:
    start_idx = content.find('<!-- START_TOMATO_AI_CHAT -->')
    end_idx = content.find('<!-- END_TOMATO_AI_CHAT -->') + len(
        '<!-- END_TOMATO_AI_CHAT -->'
    )
    content = content[:start_idx] + content[end_idx:]


WIDGET_CODE = """
<!-- START_TOMATO_AI_CHAT -->

<style>
  /* ==============================
     FLOATING CHAT BUTTON
  ============================== */

  #tv-chat-fab {
    position: fixed !important;
    bottom: 20px !important;
    right: 20px !important;
    z-index: 999999 !important;

    background: linear-gradient(135deg, #e53935, #b71c1c) !important;
    color: #ffffff !important;

    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 999px !important;

    padding: 12px 18px !important;

    font-weight: 700 !important;
    font-size: 13px !important;

    box-shadow: 0 6px 20px rgba(229, 57, 53, 0.30) !important;

    cursor: pointer !important;

    display: flex !important;
    align-items: center !important;
    gap: 8px !important;

    transition: all 0.2s ease !important;
  }

  #tv-chat-fab:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(229, 57, 53, 0.40) !important;
  }


  /* ==============================
     CHAT PANEL - COMPACT VERSION
  ============================== */

  #tv-chat-panel {
    position: fixed !important;

    bottom: 82px !important;
    right: 20px !important;

    /* Ukuran popup lebih kecil */
    width: 350px !important;
    height: 480px !important;

    /* Responsif */
    max-width: calc(100vw - 32px) !important;
    max-height: calc(100vh - 110px) !important;

    background: #ffffff !important;

    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;

    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18) !important;

    z-index: 999999 !important;

    display: none;
    flex-direction: column !important;

    overflow: hidden !important;

    font-family: system-ui, -apple-system, sans-serif !important;
  }

  #tv-chat-panel.open {
    display: flex !important;
  }


  /* ==============================
     CHAT HEADER
  ============================== */

  .tv-chat-header {
    background: linear-gradient(135deg, #d32f2f, #9a0007) !important;

    color: #ffffff !important;

    padding: 11px 13px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;

    flex-shrink: 0 !important;
  }

  .tv-chat-title-wrap {
    display: flex !important;
    align-items: center !important;
    gap: 9px !important;
  }

  .tv-chat-avatar {
    width: 34px !important;
    height: 34px !important;

    border-radius: 50% !important;

    background: rgba(255,255,255,0.20) !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    font-size: 18px !important;

    flex-shrink: 0 !important;
  }

  .tv-chat-title {
    font-weight: 700 !important;
    font-size: 13px !important;

    margin: 0 !important;

    color: #ffffff !important;
  }

  .tv-chat-subtitle {
    font-size: 10.5px !important;

    opacity: 0.9 !important;

    margin: 2px 0 0 0 !important;

    color: #ffdada !important;
  }

  .tv-chat-close {
    background: transparent !important;
    border: none !important;

    color: #ffffff !important;

    font-size: 22px !important;

    cursor: pointer !important;

    line-height: 1 !important;

    padding: 2px 5px !important;
  }


  /* ==============================
     QUICK PROMPT
  ============================== */

  .tv-chat-chips {
    padding: 8px 10px !important;

    background: #f8fafc !important;

    border-bottom: 1px solid #e2e8f0 !important;

    display: flex !important;

    gap: 6px !important;

    overflow-x: auto !important;

    white-space: nowrap !important;

    flex-shrink: 0 !important;

    scrollbar-width: thin !important;
  }

  .tv-chip {
    background: #ffffff !important;

    border: 1px solid #cbd5e1 !important;

    color: #334155 !important;

    border-radius: 999px !important;

    padding: 5px 9px !important;

    font-size: 10.5px !important;
    font-weight: 600 !important;

    cursor: pointer !important;

    flex-shrink: 0 !important;

    transition: all 0.2s ease !important;
  }

  .tv-chip:hover {
    background: #fef2f2 !important;

    border-color: #ef4444 !important;

    color: #dc2626 !important;
  }


  /* ==============================
     CHAT MESSAGES
  ============================== */

  #tv-chat-messages {
    flex: 1 !important;

    min-height: 0 !important;

    padding: 12px !important;

    overflow-y: auto !important;

    background: #f8fafc !important;

    display: flex !important;

    flex-direction: column !important;

    gap: 9px !important;

    scrollbar-width: thin !important;
  }

  .tv-msg {
    max-width: 88% !important;

    padding: 9px 11px !important;

    border-radius: 12px !important;

    font-size: 12.5px !important;

    line-height: 1.5 !important;

    word-wrap: break-word !important;

    overflow-wrap: anywhere !important;

    white-space: pre-line !important;
  }

  .tv-msg.bot {
    background: #ffffff !important;

    color: #1e293b !important;

    align-self: flex-start !important;

    border: 1px solid #e2e8f0 !important;
  }

  .tv-msg.user {
    background: #e53935 !important;

    color: #ffffff !important;

    align-self: flex-end !important;
  }


  /* ==============================
     INPUT AREA
  ============================== */

  .tv-chat-input-area {
    padding: 9px 10px !important;

    background: #ffffff !important;

    border-top: 1px solid #e2e8f0 !important;

    display: flex !important;

    gap: 7px !important;

    flex-shrink: 0 !important;
  }

  #tv-chat-input {
    flex: 1 !important;

    min-width: 0 !important;

    border: 1px solid #cbd5e1 !important;

    border-radius: 9px !important;

    padding: 9px 10px !important;

    font-size: 12px !important;

    outline: none !important;

    background: #ffffff !important;

    color: #1e293b !important;
  }

  #tv-chat-input:focus {
    border-color: #e53935 !important;

    box-shadow: 0 0 0 2px rgba(229, 57, 53, 0.08) !important;
  }

  #tv-chat-send {
    background: #e53935 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 9px !important;

    padding: 0 13px !important;

    font-weight: 700 !important;

    font-size: 12px !important;

    cursor: pointer !important;

    flex-shrink: 0 !important;

    transition: background 0.2s ease !important;
  }

  #tv-chat-send:hover {
    background: #c62828 !important;
  }


  /* ==============================
     MOBILE RESPONSIVE
  ============================== */

  @media (max-width: 480px) {

    #tv-chat-panel {
      width: calc(100vw - 24px) !important;

      height: min(520px, calc(100dvh - 100px)) !important;

      max-width: none !important;

      max-height: none !important;

      right: 12px !important;

      bottom: 76px !important;

      border-radius: 15px !important;
    }

    #tv-chat-fab {
      right: 12px !important;

      bottom: 12px !important;

      padding: 11px 15px !important;

      font-size: 12px !important;
    }

    .tv-msg {
      max-width: 92% !important;
    }
  }

</style>


<!-- FLOATING BUTTON -->

<button id="tv-chat-fab" onclick="toggleTomatoChat()">
  <span>🍅</span>
  <span>Chat AI Spesialis Tomat</span>
</button>


<!-- CHAT PANEL -->

<div id="tv-chat-panel">

  <!-- HEADER -->

  <div class="tv-chat-header">

    <div class="tv-chat-title-wrap">

      <div class="tv-chat-avatar">
        🍅
      </div>

      <div>
        <p class="tv-chat-title">
          Dr. Lycopersicon AI
        </p>

        <p class="tv-chat-subtitle">
          Spesialis Agronomi & Kematangan Tomat
        </p>
      </div>

    </div>

    <button
      class="tv-chat-close"
      onclick="toggleTomatoChat()"
      aria-label="Tutup chat"
    >
      &times;
    </button>

  </div>


  <!-- QUICK PROMPT -->

  <div class="tv-chat-chips">

    <button
      class="tv-chip"
      onclick="sendQuickPrompt('Analisis hasil deteksi fotoku barusan')"
    >
      📊 Analisis Scan
    </button>

    <button
      class="tv-chip"
      onclick="sendQuickPrompt('Jelaskan 3 fase kematangan tomat')"
    >
      🍅 Fase Kematangan
    </button>

    <button
      class="tv-chip"
      onclick="sendQuickPrompt('Cara menyimpan tomat agar awet')"
    >
      ❄️ Cara Simpan
    </button>

    <button
      class="tv-chip"
      onclick="sendQuickPrompt('Diagnosa penyakit tanaman tomat di kebun')"
    >
      🌿 Penyakit Tomat
    </button>

    <button
      class="tv-chip"
      onclick="sendQuickPrompt('Tips pemupukan tomat agar berbuah lebat')"
    >
      🌱 Pupuk & Kebun
    </button>

  </div>


  <!-- MESSAGES -->

  <div id="tv-chat-messages">

    <div class="tv-msg bot">
      Halo! Aku <b>Dr. Lycopersicon</b>, AI Spesialis Tomatmu 🍅.

      Silakan tanya seputar hasil scan kematangan tomatmu, cara penyimpanan pasca-panen, pemupukan, atau penyakit tanaman tomat di kebun!
    </div>

  </div>


  <!-- INPUT -->

  <div class="tv-chat-input-area">

    <input
      type="text"
      id="tv-chat-input"
      placeholder="Ketik pertanyaan seputar tomat..."
      onkeydown="if(event.key==='Enter') sendTomatoChat()"
    />

    <button
      id="tv-chat-send"
      onclick="sendTomatoChat()"
    >
      Kirim
    </button>

  </div>

</div>


<script>

  function toggleTomatoChat() {

    const panel = document.getElementById('tv-chat-panel');

    panel.classList.toggle('open');

    if (panel.classList.contains('open')) {

      document.getElementById('tv-chat-input').focus();

    }

  }


  function formatBoldMarkdown(text) {

    return text.replace(/\\*\\*(.*?)\\*\\*/g, '<b>$1</b>');

  }


  function appendChatMessage(text, sender) {

    const box = document.getElementById('tv-chat-messages');

    const div = document.createElement('div');

    div.className = 'tv-msg ' + sender;

    div.innerHTML = formatBoldMarkdown(text);

    box.appendChild(div);

    box.scrollTop = box.scrollHeight;

    return div;

  }


  function sendQuickPrompt(promptText) {

    document.getElementById('tv-chat-input').value = promptText;

    sendTomatoChat();

  }


  async function sendTomatoChat() {

    const input = document.getElementById('tv-chat-input');

    const msg = input.value.trim();

    if (!msg) return;

    appendChatMessage(msg, 'user');

    input.value = '';

    const loadingDiv = appendChatMessage(
      'Sedang menganalisis...',
      'bot'
    );

    try {

      const res = await fetch('/chat_ai', {

        method: 'POST',

        headers: {
          'Content-Type': 'application/json'
        },

        body: JSON.stringify({
          message: msg
        })

      });

      if (!res.ok) {
        throw new Error('HTTP ' + res.status);
      }

      const data = await res.json();

      loadingDiv.innerHTML = formatBoldMarkdown(
        data.reply || 'Maaf, terjadi gangguan.'
      );

    } catch (err) {

      loadingDiv.textContent =
        'Terjadi gangguan koneksi. Pastikan server aplikasi berjalan dan endpoint /chat_ai tersedia.';

    }

    const box = document.getElementById('tv-chat-messages');

    box.scrollTop = box.scrollHeight;

  }

</script>

<!-- END_TOMATO_AI_CHAT -->
"""


# Pasang widget sebelum tag body penutup
if '</body>' in content:

    content = content.replace(
        '</body>',
        WIDGET_CODE + '\\n</body>'
    )

elif '</BODY>' in content:

    content = content.replace(
        '</BODY>',
        WIDGET_CODE + '\\n</BODY>'
    )

else:

    content = content + '\\n' + WIDGET_CODE


# Simpan hasil
with open(html_path, 'w', encoding='utf-8') as f:

    f.write(content)


print(
    "[BERHASIL] Widget Chat AI Spesialis Tomat "
    "versi compact berhasil dipasang!"
)