import streamlit as st
import numpy as np
import time
import random
import os

from live_predict2 import VoiceCommandRecognizer 


if "board" not in st.session_state:
    st.session_state.board = [" "] * 9
if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = 4  
if "current_player" not in st.session_state:
    st.session_state.current_player = "X" 
if "last_input_time" not in st.session_state:
    st.session_state.last_input_time = time.time()


input_interval = 3.0


MODEL_DIR = "/Users/mariushorn/Desktop/hvl/6_semester/DAT255/Eksamensoppgave/DAT255-VoiceCommandAssistant/models"
recognizer = VoiceCommandRecognizer(MODEL_DIR)


def move_cursor(selected_cell, direction):
    row = selected_cell // 3
    col = selected_cell % 3
    if direction == "up" and row > 0:
        row -= 1
    elif direction == "down" and row < 2:
        row += 1
    elif direction == "left" and col > 0:
        col -= 1
    elif direction == "right" and col < 2:
        col += 1
    return row * 3 + col

def check_winner(board):
    wins = [
        [0,1,2], [3,4,5], [6,7,8],   # rows
        [0,3,6], [1,4,7], [2,5,8],   # cols
        [0,4,8], [2,4,6]             # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] and board[a] != " ":
            return board[a]
    if " " not in board:
        return "Draw"
    return None

def format_board_html(board, selected_cell):
 
    html = """
    <style>
    .tictactoe {
      border-collapse: collapse;
      margin: 0 auto;
      table-layout: fixed;
      width: 210px; /* 3 columns * 70px */
    }
    .tictactoe td {
      border: 2px solid #ccc;
      width: 70px; height: 70px;
      text-align: center; vertical-align: middle;
      font-size: 28px; font-weight: bold;
      font-family: sans-serif;
    }
    .selected { background-color: #FFE066; }
    .x-cell    { color: #FF3333; }
    .o-cell    { color: #3333FF; }
    </style>
    <table class="tictactoe">
    """
    for r in range(3):
        html += "<tr>"
        for c in range(3):
            i = r*3 + c
            val = board[i]
            classes = []
            if i == selected_cell:
                classes.append("selected")
            if val == "X":
                classes.append("x-cell")
            elif val == "O":
                classes.append("o-cell")
            cls = " ".join(classes)
            disp = val if val != " " else ""
            html += f'<td class="{cls}">{disp}</td>'
        html += "</tr>"
    html += "</table>"
    return html

# ====================== Streamlit UI ======================
st.title("Voice-Controlled Tic Tac Toe")

st.markdown(
   
)

# Initial board render
board_html = format_board_html(
    st.session_state.board,
    st.session_state.selected_cell
)
board_placeholder = st.empty()
board_placeholder.markdown(board_html, unsafe_allow_html=True)

if st.button("Speak Command"):
    now = time.time()
    if now - st.session_state.last_input_time >= input_interval:
        st.session_state.last_input_time = now

        st.info("Listening...")
        command = recognizer.listen_and_predict()
        st.success(f"Predicted command: **{command}**")

        # move cursor or place mark
        if command in ["up","down","left","right"]:
            st.session_state.selected_cell = move_cursor(
                st.session_state.selected_cell, command
            )
        elif command == "yes":
            idx = st.session_state.selected_cell
            if st.session_state.board[idx] == " ":
                # place X
                st.session_state.board[idx] = st.session_state.current_player
                winner = check_winner(st.session_state.board)
                if winner:
                    # —— special end-game effects! ——
                    if winner == "Draw":
                        st.warning("Game Over: Draw!")
                        st.snow()
                    elif winner == "X":
                        st.success("You win!")
                        st.balloons()
                    else:  # O wins
                        st.error("Oh no! You lost.")
                        st.markdown(
                            """
                            <script>
                              const b = document.body;
                              let t = 0;
                              const shake = setInterval(() => {
                                b.style.transform = `translate(${(Math.random()-0.5)*10}px, ${(Math.random()-0.5)*10}px)`;
                                if (++t > 15) {
                                  clearInterval(shake);
                                  b.style.transform = '';
                                }
                              }, 50);
                            </script>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    # O’s random move
                    free = [i for i, v in enumerate(st.session_state.board) if v == " "]
                    if free:
                        mv = random.choice(free)
                        st.session_state.board[mv] = "O"
                        st.session_state.current_player = "X"
                        w = check_winner(st.session_state.board)
                        if w:
                            if w == "Draw":
                                st.warning("Game Over: Draw!")
                                st.snow()
                            elif w == "X":
                                st.success("You win!")
                                st.balloons()
                            else:
                                st.error(" Oh no! You lost. ")
                                st.markdown(
                                    """
                                    <script>
                                      const b = document.body;
                                      let t = 0;
                                      const shake = setInterval(() => {
                                        b.style.transform = `translate(${(Math.random()-0.5)*10}px, ${(Math.random()-0.5)*10}px)`;
                                        if (++t > 15) {
                                          clearInterval(shake);
                                          b.style.transform = '';
                                        }
                                      }, 50);
                                    </script>
                                    """,
                                    unsafe_allow_html=True,
                                )
            else:
                st.warning("Cell taken—try another.")
        else:
            st.warning("Unrecognized command.")

        # re-render board
        board_html = format_board_html(
            st.session_state.board,
            st.session_state.selected_cell
        )
        board_placeholder.markdown(board_html, unsafe_allow_html=True)
    else:
        st.warning("Please wait a few seconds before speaking again.")
