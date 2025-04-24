import pygame
import time
from live_predict import record_audio, predict_command

class TicTacToeGame:
    def __init__(self, width=300, height=300):
        pygame.init()
        self.selected_cell = 4  # Start in the center
        self.width = width
        self.height = height
        self.line_width = 5  # Bredden på linjene
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tic Tac Toe")
        self.board = [" "] * 9  # 3x3 brett representert som en liste med 9 elementer
        self.current_player = "X"

    def draw_board(self):
        self.screen.fill((255, 255, 255))
        
        # Tegn vertikale linjer
        pygame.draw.line(self.screen, (0, 0, 0), (self.width // 3, 0), (self.width // 3, self.height), self.line_width)
        pygame.draw.line(self.screen, (0, 0, 0), (2 * self.width // 3, 0), (2 * self.width // 3, self.height), self.line_width)
        
        # Tegn horisontale linjer
        pygame.draw.line(self.screen, (0, 0, 0), (0, self.height // 3), (self.width, self.height // 3), self.line_width)
        pygame.draw.line(self.screen, (0, 0, 0), (0, 2 * self.height // 3), (self.width, 2 * self.height // 3), self.line_width)
        
        # Tegn X og O på brettet
        for i in range(9):
            row = i // 3
            col = i % 3
            x = col * (self.width // 3) + (self.width // 6)
            y = row * (self.height // 3) + (self.height // 6)
            if self.board[i] == "X":
                pygame.draw.line(self.screen, (255, 0, 0), (x - 30, y - 30), (x + 30, y + 30), self.line_width)
                pygame.draw.line(self.screen, (255, 0, 0), (x - 30, y + 30), (x + 30, y - 30), self.line_width)
            elif self.board[i] == "O":
                pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 30, self.line_width)
        pygame.display.update()

        # Highlight selected cell
        row = self.selected_cell // 3
        col = self.selected_cell % 3
        x = col * (self.width // 3)
        y = row * (self.height // 3)
        cell_rect = pygame.Rect(x, y, self.width // 3, self.height // 3)
        pygame.draw.rect(self.screen, (0, 255, 0), cell_rect, 3)




    def update_board(self, pos):
        if self.board[pos] == " ":
            self.board[pos] = self.current_player
            self.current_player = "O" if self.current_player == "X" else "X"
    
    def check_winner(self):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rader
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # kolonner
            [0, 4, 8], [2, 4, 6]              # diagonaler
        ]
        for condition in win_conditions:
            a, b, c = condition
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != " ":
                return self.board[a]
        if " " not in self.board:
            return "Draw"
        return None

    def run(self):
        last_input_time = 0
        running = True
        clock = pygame.time.Clock()
        self.draw_board()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // (self.width // 3)
                    row = y // (self.height // 3)
                    pos = row * 3 + col
                    if self.board[pos] == " ":
                        self.update_board(pos)
                        self.draw_board()
                        winner = self.check_winner()
                        if winner:
                            print(f"Winner: {winner}")
                            running = False
            clock.tick(30)
        pygame.quit()
