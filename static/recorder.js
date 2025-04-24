// static/recorder.js

class VoiceTicTacToe {
    constructor(recordButtonId, messageDivId, boardDivId) {
      this.btn     = document.getElementById(recordButtonId);
      this.msg     = document.getElementById(messageDivId);
      this.board   = document.getElementById(boardDivId);
      this.recDur  = 1500; // ms
      this.chunks  = [];
      this.recorder = null;
      this._init();
    }
  
    _init() {
      this.btn.addEventListener('click', () => this._onClick());
    }
  
    async _onClick() {
      this.btn.disabled = true;
      try {
        await this._record();
        this.btn.textContent = '⏳ Sending…';
        const blob = new Blob(this.chunks, { type: 'audio/webm' });
        await this._send(blob);
      } catch (err) {
        console.error(err);
        this.msg.textContent = 'Error accessing microphone or server.';
      } finally {
        this.btn.textContent = '🎙️ Speak Command';
        this.btn.disabled = false;
      }
    }
  
    async _record() {
      this.msg.textContent = '🔴 Recording…';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.recorder = new MediaRecorder(stream);
      this.chunks = [];
      this.recorder.ondataavailable = e => this.chunks.push(e.data);
      this.recorder.start();
  
      return new Promise(resolve => {
        setTimeout(() => {
          this.recorder.stop();
          this.recorder.onstop = () => resolve();
        }, this.recDur);
      });
    }
  
    async _send(blob) {
      let res, data;
      try {
        res = await fetch('/play', {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: await blob.arrayBuffer()
        });
        data = await res.json();
      } catch (err) {
        throw new Error('Network error');
      }
  
      if (!res.ok) {
        this.msg.textContent = data.error || 'Server error';
        return;
      }
  
      // success!
      this.board.innerHTML = data.board_html;
      this.msg.textContent   = data.message;
    }
  }
  
  // wire it up once DOM is ready
  window.addEventListener('DOMContentLoaded', () => {
    new VoiceTicTacToe('rec-btn', 'msg', 'board');
  });
  