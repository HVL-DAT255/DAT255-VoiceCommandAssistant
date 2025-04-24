import gradio as gr
import numpy as np
import random
import librosa
from live_predict2 import VoiceCommandRecognizer

recognizer    = VoiceCommandRecognizer("models")
INPUT_INTERVAL = 3.0

def init_state():
    return {"board":[" "]*9, "sel":4, "last":0}

def move_cursor(sel, d):
    r,c = divmod(sel,3)
    if d=="up"    and r>0:    r-=1
    if d=="down"  and r<2:    r+=1
    if d=="left"  and c>0:    c-=1
    if d=="right" and c<2:    c+=1
    return r*3+c

def format_board(b, sel):
    # (same HTML/CSS you already have)
    html = "<table style='border-collapse:collapse;margin:0 auto;'>"
    for r in range(3):
        html += "<tr>"
        for c in range(3):
            i = r*3+c
            style = "width:70px;height:70px;border:2px solid #ccc;"
            if i==sel: style+="background:#FFE066;"
            color = ""
            if b[i]=="X": color="color:#FF3333;"
            if b[i]=="O": color="color:#3333FF;"
            html += f"<td style='{style}{color}font-size:28px;text-align:center;'>{b[i] if b[i]!=" " else ""}</td>"
        html += "</tr>"
    return html+"</table>"

def step(audio, state):
    import time
    now = time.time()
    if now - state["last"] < INPUT_INTERVAL:
        return format_board(state["board"],state["sel"]), "⏳ Please wait…", state
    state["last"] = now

    # load & resample to recognizer.SAMPLE_RATE
    y, sr = librosa.load(audio.name, sr=recognizer.SAMPLE_RATE)
    cmd   = recognizer.predict(y.astype("float32"))

    msg = ""
    if cmd in ["up","down","left","right"]:
        state["sel"] = move_cursor(state["sel"], cmd)
        msg = f"Moved {cmd}"
    elif cmd=="yes":
        idx = state["sel"]
        if state["board"][idx]==" ":
            state["board"][idx] = "X"
            # check X win / draw…
            free = [i for i,v in enumerate(state["board"]) if v==" "]
            if free:
                mv = random.choice(free)
                state["board"][mv] = "O"
                msg = "You played X; computer moved"
            else:
                msg = "Game Over"
        else:
            msg = "Cell taken"
    else:
        msg = "Unrecognized command"

    return format_board(state["board"],state["sel"]), msg, state

with gr.Blocks() as demo:
    state = gr.State(init_state())
    board = gr.HTML(format_board(state.value["board"],state.value["sel"]))
    status = gr.Text()
    mic    = gr.Audio(source="microphone", type="filepath")
    btn    = gr.Button("🎙️ Speak Command")

    btn.click(fn=step,
              inputs=[mic, state],
              outputs=[board, status, state])

demo.launch()
