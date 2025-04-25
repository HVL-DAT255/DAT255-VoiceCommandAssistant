import pygame
import time
from game import TicTacToeGame
from live_predict import record_audio, predict_command

class VoiceControlledGame(TicTacToeGame):
    def __init__(self, width=300, height=300):
        super().__init__(width, height)
        self.selected_cell = 4

    def move_cursor(self, direction):
        row = self.selected_cell // 3
        col = self.selected_cell % 3

        if direction == "up" and row > 0:
            row -= 1
        elif direction == "down" and row < 2:
            row += 1
        elif direction == "left" and col > 0:
            col -= 1
        elif direction == "right" and col < 2:
            col += 1

        self.selected_cell = row * 3 + col

    def draw_board(self):
        super().draw_board()
        # Highlight selected cell
        row = self.selected_cell // 3
        col = self.selected_cell % 3
        x = col * (self.width // 3)
        y = row * (self.height // 3)
        cell_rect = pygame.Rect(x, y, self.width // 3, self.height // 3)
        pygame.draw.rect(self.screen, (0, 255, 0), cell_rect, 3)
        pygame.display.update()

    def run(self):
        running = True
        clock = pygame.time.Clock()
        last_input_time = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if time.time() - last_input_time > 3.0:
                self.draw_board()
                audio = record_audio()
                command = predict_command(audio)
                print(f"Predicted command: {command}")
                last_input_time = time.time()

                if command in ["up", "down", "left", "right"]:
                    self.move_cursor(command)
                elif command == "yes":
                    if self.board[self.selected_cell] == " ":
                        self.update_board(self.selected_cell)
                        winner = self.check_winner()
                        if winner:
                            if winner == "Draw":
                                print("It's a draw!")
                            else:
                                print(f"{winner} wins!")
                            time.sleep(2)
                            running = False

            clock.tick(30)  # keep frame rate consistent

        pygame.quit()


if __name__ == "__main__":
    game = VoiceControlledGame()
    game.run()
