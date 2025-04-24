@app.route('/')
def index():
    return render_template('index.html', board_html=format_board_html(board, selected_cell))

@app.route('/play', methods=['POST'])
def play():
    global last_input_time, board, selected_cell, token
    now = time.time()
    if now - last_input_time < INPUT_INTERVAL:
        return jsonify({'error':'wait'})
    last_input_time = now

    wav = request.data
    audio, sr = sf.read(io.BytesIO(wav))
    if sr != recognizer.SAMPLE_RATE:
        audio = librosa.resample(audio, sr, recognizer.SAMPLE_RATE)
    cmd = recognizer.predict(audio.astype('float32'))

    msg = ''
    if cmd in ['up','down','left','right']:
        selected_cell = move_cursor(selected_cell, cmd)
        msg = f'Moved {cmd}'
    elif cmd=='yes':
        if board[selected_cell]==' ': board[selected_cell]=token
        else: msg='Cell taken'
        win = check_winner(board)
        if win:
            msg = f'Game Over: {win}'
        else:
            free = [i for i,v in enumerate(board) if v==' ']
            mv = random.choice(free) if free else None
            if mv is not None: board[mv]='O'; token='X'; msg='Computer moved'
    else:
        msg='Unknown'

    return jsonify({'command':cmd,'board_html':format_board_html(board,selected_cell),'message':msg})

if __name__=='__main__':
    app.run(debug=True)