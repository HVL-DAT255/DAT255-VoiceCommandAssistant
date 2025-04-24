import os
import io
import time
import random
from flask import Flask, request, render_template, jsonify
from io import BytesIO
import numpy as np
import librosa
from pydub import AudioSegment
from live_predict2 import VoiceCommandRecognizer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_INTERVAL = 3.0
MODEL_DIR      = "models"     # adjust if your models live elsewhere
PORT           = 5000

# ── FLASK SETUP ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
recognizer = VoiceCommandRecognizer(MODEL_DIR)

# ── GAME STATE ──────────────────────────────────────────────────────────────────
board          = [" "] * 9
selected_cell  = 4
current_player = "X"  # human is X, computer O
last_input_time= 0.0
# ── HELPERS ─────────────────────────────────────────────────────────────────────
def move_cursor(cell, direction):
    r, c = divmod(cell, 3)
    if direction=="up"    and r>0: r-=1
    if direction=="down"  and r<2: r+=1
    if direction=="left"  and c>0: c-=1
    if direction=="right" and c<2: c+=1
    return r*3 + c

def check_winner(b):
    wins = ([0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6])
    for a,b1,c in wins:
        if b[a]==b[b1]==b[c]!=" ": return b[a]
    return "Draw" if " " not in b else None

def format_board_html(b, sel):
    # your existing CSS + HTML formatter...
    html = """
    <style>
      .tictactoe{border-collapse:collapse;margin:0 auto;width:210px;}
      .tictactoe td{border:2px solid #ccc;width:70px;height:70px;text-align:center;
        vertical-align:middle;font-size:28px;font-weight:bold;font-family:sans-serif;}
      .selected{background-color:#FFE066;}
      .x-cell{color:#FF3333;} .o-cell{color:#3333FF;}
    </style>
    <table class="tictactoe">"""
    for r in range(3):
        html += "<tr>"
        for c in range(3):
            i = r*3 + c
            cls = []
            if i==sel:       cls.append("selected")
            if b[i]=="X":    cls.append("x-cell")
            elif b[i]=="O":  cls.append("o-cell")
            disp = b[i] if b[i]!=" " else ""
            html += f'<td class="{" ".join(cls)}">{disp}</td>'
        html += "</tr>"
    html += "</table>"
    return html

# ── ROUTES ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           board_html=format_board_html(board, selected_cell))

@app.route("/play", methods=["POST"])
def play():
    global last_input_time, board, selected_cell, current_player

    # rate-limit checks
    now = time.time()
    if now - last_input_time < INPUT_INTERVAL:
        return jsonify({"error":"wait"}), 429
    last_input_time = now

    # 1) Read incoming bytes
    wav_bytes = request.get_data()

    try:
        # 2) Decode WebM/Opus → raw PCM via pydub/ffmpeg
        aud = AudioSegment.from_file(BytesIO(wav_bytes), format="webm")
    except Exception as e:
        return jsonify({"error":"decode failed"}), 400

    # 3) Extract samples, mono, float32
    samples = np.array(aud.get_array_of_samples())
    if aud.channels > 1:
        samples = samples.reshape(-1, aud.channels).mean(axis=1)
    audio = (samples / np.iinfo(samples.dtype).max).astype("float32")

    # 4) Resample if needed
    if aud.frame_rate != recognizer.SAMPLE_RATE:
        audio = librosa.resample(audio,
                                 orig_sr=aud.frame_rate,
                                 target_sr=recognizer.SAMPLE_RATE)

    # 5) Predict command
    cmd = recognizer.predict(audio)

    # 6) Game logic
    msg = ""
    if cmd in ["up","down","left","right"]:
        selected_cell = move_cursor(selected_cell, cmd)
        msg = f"Moved {cmd}"
    elif cmd=="yes":
        if board[selected_cell]==" ":
            board[selected_cell] = current_player
            win = check_winner(board)
            if win:
                msg = f"Game Over: {win}"
            else:
                # computer move
                free = [i for i,v in enumerate(board) if v==" "]
                if free:
                    mv = random.choice(free)
                    board[mv] = "O"
                    msg = "Computer moved"
        else:
            msg = "Cell taken"
    else:
        msg = "Unknown command"

    return jsonify({
        "command": cmd,
        "board_html": format_board_html(board, selected_cell),
        "message": msg
    })

# ── LAUNCH ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=PORT)
