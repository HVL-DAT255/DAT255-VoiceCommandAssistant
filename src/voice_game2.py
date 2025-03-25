import pygame
import time
import random
from game import TicTacToeGame
from live_predict import record_audio, predict_command

class VoiceControlledGame(TicTacToeGame):
    def __init__(self, width=300, height=300):
        super().__init__(width, height)
        # Initialize the pointer (cursor) at the center cell (index 4)
        self.selected_cell = 4
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.last_input_time = time.time()
        self.input_interval = 3.0  # Wait 5 seconds between commands
        self.waiting_for_input = True

    def draw_message(self, text, color=(0, 0, 0)):
        """Display a message at the bottom of the screen."""
        message_surface = self.font.render(text, True, color)
        rect = message_surface.get_rect(center=(self.width // 2, self.height - 20))
        # Draw a white rectangle to clear the previous message area
        clear_rect = pygame.Rect(0, self.height - 40, self.width, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), clear_rect)
        self.screen.blit(message_surface, rect)
        pygame.display.update()

    def move_cursor(self, direction):
        """Update pointer location based on a directional command."""
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
        """Draw the board (using the base TicTacToeGame drawing) and highlight the selected cell."""
        super().draw_board()
        row = self.selected_cell // 3
        col = self.selected_cell % 3
        x = col * (self.width // 3)
        y = row * (self.height // 3)
        pointer_rect = pygame.Rect(x, y, self.width // 3, self.height // 3)
        pygame.draw.rect(self.screen, (0, 255, 0), pointer_rect, 3)
        pygame.display.update()

    def run(self):
        running = True
        clock = pygame.time.Clock()
        # Show initial instructions on screen
        self.draw_board()
        self.draw_message("Press SPACE to start voice commands")
        
        # Wait for user to press SPACE to start the game loop
        waiting_for_start = True
        while waiting_for_start:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting_for_start = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    waiting_for_start = False
            clock.tick(30)

        # Main game loop
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Check if it's time to listen for a command
            if time.time() - self.last_input_time > self.input_interval:
                self.draw_board()
                self.draw_message("Listening...", color=(0, 0, 255))
                command = predict_command(record_audio())
                self.draw_message(f"Command: {command}", color=(0, 0, 0))
                print("Predicted command:", command)
                self.last_input_time = time.time()

                if command in ["up", "down", "left", "right"]:
                    self.move_cursor(command)
                elif command == "yes":
                    if self.board[self.selected_cell] == " ":
                        self.update_board(self.selected_cell)
                        winner = self.check_winner()
                        if winner:
                            if winner == "Draw":
                                self.draw_message("It's a draw!", color=(255, 0, 0))
                                print("It's a draw!")
                            else:
                                self.draw_message(f"{winner} wins!", color=(255, 0, 0))
                                print(f"{winner} wins!")
                            time.sleep(2)
                            running = False
                        else:
                            self.current_player = "O"  # Switch turn after human move
                    else:
                        self.draw_message("Cell taken! Try again.", color=(255, 0, 0))
                else:
                    self.draw_message("Unrecognized command.", color=(255, 0, 0))

                self.draw_board()

                # Computer's turn: choose a random available cell
                if self.current_player == "O":
                    pygame.time.wait(500)
                    available = [i for i, cell in enumerate(self.board) if cell == " "]
                    if available:
                        move = random.choice(available)
                        self.board[move] = self.current_player
                        self.current_player = "X"
                        winner = self.check_winner()
                        if winner:
                            if winner == "Draw":
                                self.draw_message("It's a draw!", color=(255, 0, 0))
                                print("It's a draw!")
                            else:
                                self.draw_message(f"{winner} wins!", color=(255, 0, 0))
                                print(f"{winner} wins!")
                            time.sleep(2)
                            running = False
                    self.draw_board()

            clock.tick(30)
        pygame.quit()

if __name__ == "__main__":
    game = VoiceControlledGame()
    game.run()
