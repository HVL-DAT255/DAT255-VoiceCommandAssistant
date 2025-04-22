import streamlit as st
import numpy as np
import time
import random
import os

from live_predict3 import record_audio, predict_command  # Your audio+ML functions

# ====================== Session State Initialization ======================
if "board" not in st.session_state:
    st.session_state.board = [" "] * 9
if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = 4  # Start in center
if "current_player" not in st.session_state:
    st.session_state.current_player = "X"  # Human is X, computer is O
if "last_input_time" not in st.session_state:
    st.session_state.last_input_time = time.time()

# Interval (in seconds) between voice commands
input_interval = 3.0

# ====================== Game Logic ======================
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
        [0,4,8], [2,4,6]            # diagonals
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] and board[a] != " ":
            return board[a]
    if " " not in board:
        return "Draw"
    return None

def format_board_html(board, selected_cell):
    """
    Return an HTML table to display the board, highlighting the selected cell
    and coloring X and O differently.
    """
    html = """
    <style>
    .tictactoe {
        border-collapse: collapse;
        margin: 0 auto;
        table-layout: fixed;
        width: 210px; /* 3 columns * 70px each */
    }
    .tictactoe td {
        border: 2px solid #ccc;
        width: 70px;
        height: 70px;
        text-align: center;
        vertical-align: middle;
        font-size: 28px;
        font-weight: bold;
        font-family: sans-serif;
    }
    .selected {
        background-color: #FFE066; /* highlight color */
    }
    .x-cell {
        color: #FF3333;  /* X in red */
    }
    .o-cell {
        color: #3333FF;  /* O in blue */
    }
    </style>
    <table class="tictactoe">
    """

    for row in range(3):
        html += "<tr>"
        for col in range(3):
            i = row * 3 + col
            cell_value = board[i]
            cell_classes = []
            # highlight selected cell
            if i == selected_cell:
                cell_classes.append("selected")
            # color X or O
            if cell_value == "X":
                cell_classes.append("x-cell")
            elif cell_value == "O":
                cell_classes.append("o-cell")

            class_str = " ".join(cell_classes)
            # blank if empty
            display_val = cell_value if cell_value != " " else ""
            html += f'<td class="{class_str}">{display_val}</td>'
        html += "</tr>"
    html += "</table>"
    return html

# ====================== Streamlit UI ======================
st.title("Voice-Controlled Tic Tac Toe")

st.markdown(
    """
    **Instructions:**
    - Press **Speak Command** to record your voice.
    - Supported commands: **up**, **down**, **left**, **right**, **yes**.
    - **"yes"** places your mark (X) on the highlighted cell.
    - The computer (O) makes a random move afterward.
    """
)

# Build and display the HTML board
board_html = format_board_html(st.session_state.board, st.session_state.selected_cell)
board_placeholder = st.empty()
board_placeholder.markdown(board_html, unsafe_allow_html=True)

if st.button("Speak Command"):
    current_time = time.time()
    if current_time - st.session_state.last_input_time >= input_interval:
        st.session_state.last_input_time = current_time

        st.info("Listening...")
        audio = record_audio()
        command = predict_command(audio)
        st.success(f"Predicted command: **{command}**")

        if command in ["up", "down", "left", "right"]:
            st.session_state.selected_cell = move_cursor(st.session_state.selected_cell, command)
        elif command == "yes":
            if st.session_state.board[st.session_state.selected_cell] == " ":
                st.session_state.board[st.session_state.selected_cell] = st.session_state.current_player
                winner = check_winner(st.session_state.board)
                if winner:
                    if winner == "Draw":
                        st.warning("Game Over! It's a draw!")
                    else:
                        st.warning(f"Game Over! {winner} wins!")
                else:
                    # Computer move
                    free_cells = [i for i, cell in enumerate(st.session_state.board) if cell == " "]
                    if free_cells:
                        move = random.choice(free_cells)
                        st.session_state.board[move] = "O"
                        st.session_state.current_player = "X"
                        w = check_winner(st.session_state.board)
                        if w:
                            if w == "Draw":
                                st.warning("Game Over! It's a draw!")
                            else:
                                st.warning(f"Game Over! {w} wins!")
            else:
                st.warning("Cell already taken. Try a different move.")
        else:
            st.warning("Unrecognized command.")

        # Update the board
        board_html = format_board_html(st.session_state.board, st.session_state.selected_cell)
        board_placeholder.markdown(board_html, unsafe_allow_html=True)
    else:
        st.warning("Please wait a bit longer before speaking again.")
