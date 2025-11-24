import tkinter as tk
from tkinter import messagebox, ttk, font
import time
from PIL import Image, ImageTk
import os
import colorsys

class CubicUI:
    def __init__(self, root, game, ai):
        self.game = game
        self.ai = ai
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Define color scheme
        self.colors = {
            "primary": "#3498db",       # Blue
            "secondary": "#2ecc71",    # Green
            "accent": "#e74c3c",       # Red
            "background": "#f5f5f5",   # Light gray
            "dark": "#2c3e50",         # Dark blue/gray
            "light": "#ecf0f1",        # Very light gray
            "text": "#34495e",         # Dark gray/blue
            "x_color": "#e74c3c",      # Red for X
            "o_color": "#3498db",      # Blue for O
            "highlight": "#2ecc71"     # Green for winning line
        }
        
        # Configure root window
        self.root.configure(bg=self.colors["background"])
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Segoe UI", size=10)
        self.root.option_add("*Font", self.default_font)
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Segoe UI", 11))
        self.style.configure("TLabel", font=("Segoe UI", 11))
        self.style.configure("TFrame", background=self.colors["background"])
        
        # Create frames
        self.menu_frame = tk.Frame(root, bg=self.colors["background"])
        self.game_frame = tk.Frame(root, bg=self.colors["background"])

        # Game mode: 'ai' or 'friend'
        self.game_mode = 'ai'
        
        # AI algorithm options
        self.ai_algorithms = [
            ("heuristic", "Heuristic Only (Default)"),
            ("minimax", "Minimax Only"),
            ("alpha_beta", "Minimax + Alpha-Beta Pruning"),
            ("symmetry", "Symmetry Reduction"),
            ("heuristic_reduction", "Heuristic Reduction"),
            ("minimax_heuristic", "Minimax + Evaluation Heuristic"),
            ("minimax_astar", "Minimax + A* Search")
        ]
        
        self.create_menu()  
        self.create_game_ui()

        self.show_menu()

        self.start_time = None
        self.timer_running = False
        
        # Track game statistics
        self.stats = {
            "x_wins": 0,
            "o_wins": 0,
            "draws": 0,
            "games_played": 0
        }

    def create_menu(self):
        self.menu_frame.pack(fill="both", expand=True)

        # Create a container for better layout
        container = tk.Frame(self.menu_frame, bg=self.colors["background"], padx=30, pady=20)
        container.pack(expand=True)
        
        # Game title with shadow effect
        title_frame = tk.Frame(container, bg=self.colors["background"])
        title_frame.pack(pady=(0, 30))
        
        # Shadow effect (slightly offset label)
        shadow_label = tk.Label(title_frame, text="3D Tic-Tac-Toe", 
                              font=("Segoe UI", 36, "bold"), 
                              fg="#aaaaaa",  # Light gray for shadow
                              bg=self.colors["background"])
        shadow_label.grid(row=0, column=0, padx=2, pady=2)
        
        # Main title
        title = tk.Label(title_frame, text="3D Tic-Tac-Toe", 
                       font=("Segoe UI", 36, "bold"), 
                       fg=self.colors["primary"],
                       bg=self.colors["background"])
        title.grid(row=0, column=0)
        
        # Subtitle
        subtitle = tk.Label(container, text="Challenge your spatial thinking", 
                          font=("Segoe UI", 14), 
                          fg=self.colors["text"],
                          bg=self.colors["background"])
        subtitle.pack(pady=(0, 30))
        
        # Mode selection frame with rounded corners and border
        mode_frame = tk.Frame(container, bg=self.colors["light"], 
                            padx=20, pady=15, bd=1, relief="solid")
        mode_frame.pack(fill="x", pady=10)
        
        mode_label = tk.Label(mode_frame, text="Select Game Mode", 
                             font=("Segoe UI", 16, "bold"),
                             fg=self.colors["dark"],
                             bg=self.colors["light"])
        mode_label.pack(pady=(0, 10), anchor="w")
        
        # Radio buttons for game mode selection with custom styling
        self.mode_var = tk.StringVar(value="ai")
        
        radio_frame = tk.Frame(mode_frame, bg=self.colors["light"])
        radio_frame.pack(fill="x")
        
        # Custom radio buttons with icons (simulated)
        ai_radio = tk.Radiobutton(radio_frame, text="Play with AI", variable=self.mode_var, 
                                value="ai", font=("Segoe UI", 12),
                                command=self.toggle_ai_options,
                                bg=self.colors["light"],
                                activebackground=self.colors["light"],
                                selectcolor=self.colors["light"],
                                fg=self.colors["text"],
                                highlightthickness=0)
        ai_radio.pack(side="left", padx=(0, 20))
        
        friend_radio = tk.Radiobutton(radio_frame, text="Play with Friend", variable=self.mode_var, 
                                   value="friend", font=("Segoe UI", 12),
                                   command=self.toggle_ai_options,
                                   bg=self.colors["light"],
                                   activebackground=self.colors["light"],
                                   selectcolor=self.colors["light"],
                                   fg=self.colors["text"],
                                   highlightthickness=0)
        friend_radio.pack(side="left")
        
        # AI algorithm selection frame
        self.ai_frame = tk.Frame(container, bg=self.colors["light"], 
                               padx=20, pady=15, bd=1, relief="solid")
        self.ai_frame.pack(fill="x", pady=10)
        
        ai_algo_label = tk.Label(self.ai_frame, text="Select AI Algorithm", 
                                font=("Segoe UI", 16, "bold"),
                                fg=self.colors["dark"],
                                bg=self.colors["light"])
        ai_algo_label.pack(pady=(0, 10), anchor="w")
        
        # Description of AI algorithms
        ai_desc = tk.Label(self.ai_frame, 
                         text="Choose the AI algorithm that will challenge you",
                         font=("Segoe UI", 10),
                         fg=self.colors["text"],
                         bg=self.colors["light"],
                         justify="left")
        ai_desc.pack(anchor="w", pady=(0, 10))
        
        # Styled dropdown for AI algorithm selection
        self.algorithm_var = tk.StringVar(value="heuristic")
        algorithm_dropdown = ttk.Combobox(self.ai_frame, textvariable=self.algorithm_var, 
                                       font=("Segoe UI", 12), state="readonly", width=30)
        algorithm_dropdown['values'] = [name for _, name in self.ai_algorithms]
        algorithm_dropdown.current(0)
        algorithm_dropdown.pack(fill="x", pady=5)

        # Button frame for better alignment
        button_frame = tk.Frame(container, bg=self.colors["background"])
        button_frame.pack(pady=20, fill="x")
        
        # Styled start button
        start_button = tk.Button(button_frame, text="Start Game", 
                               font=("Segoe UI", 14, "bold"),
                               bg=self.colors["primary"], fg="white",
                               activebackground=self.colors["primary"],  # Slightly transparent on hover
                               activeforeground="white",
                               relief="flat", bd=0,
                               padx=20, pady=10,
                               command=self.start_game)
        start_button.pack(side="left", expand=True, fill="x", padx=(0, 5))

        # Styled close button
        close_button = tk.Button(button_frame, text="Exit", 
                               font=("Segoe UI", 14, "bold"),
                               bg=self.colors["dark"], fg="white",
                               activebackground=self.colors["dark"],
                               activeforeground="white",
                               relief="flat", bd=0,
                               padx=20, pady=10,
                               command=self.on_close)
        close_button.pack(side="right", expand=True, fill="x", padx=(5, 0))
        
        # Add version info at the bottom
        version_label = tk.Label(self.menu_frame, text="v1.0.0", 
                               font=("Segoe UI", 8), 
                               fg="#999999",  # Semi-transparent
                               bg=self.colors["background"])
        version_label.pack(side="bottom", pady=5)

    def create_game_ui(self):
        self.game_frame.pack_forget()

        # Main container for game UI
        game_container = tk.Frame(self.game_frame, bg=self.colors["background"])
        game_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Header with game info
        header_frame = tk.Frame(game_container, bg=self.colors["background"])
        header_frame.pack(fill="x", pady=(0, 15))
        
        # Game title in the header
        header_title = tk.Label(header_frame, text="3D Tic-Tac-Toe", 
                              font=("Segoe UI", 18, "bold"), 
                              fg=self.colors["primary"],
                              bg=self.colors["background"])
        header_title.pack(side="left")
        
        # Status panel with modern styling
        status_panel = tk.Frame(game_container, bg=self.colors["light"], bd=1, relief="solid")
        status_panel.pack(fill="x", pady=(0, 15))
        
        status_inner = tk.Frame(status_panel, bg=self.colors["light"], padx=15, pady=10)
        status_inner.pack(fill="x")
        
        # Status label with icon indicator
        status_container = tk.Frame(status_inner, bg=self.colors["light"])
        status_container.pack(side="left")
        
        # Player indicator (colored circle)
        self.player_indicator = tk.Canvas(status_container, width=20, height=20, 
                                       bg=self.colors["light"], highlightthickness=0)
        self.player_indicator.pack(side="left", padx=(0, 10))
        self.player_indicator.create_oval(2, 2, 18, 18, fill=self.colors["x_color"], outline="")
        
        # Status text
        self.status_label = tk.Label(status_container, text="Player X's Turn", 
                                   font=("Segoe UI", 14, "bold"),
                                   fg=self.colors["text"],
                                   bg=self.colors["light"])
        self.status_label.pack(side="left")

        # Timer with icon
        timer_container = tk.Frame(status_inner, bg=self.colors["light"])
        timer_container.pack(side="right")
        
        timer_icon = tk.Label(timer_container, text="⏱", font=("Segoe UI", 14),
                            bg=self.colors["light"])
        timer_icon.pack(side="left", padx=(0, 5))
        
        self.timer_label = tk.Label(timer_container, text="00:00", 
                                  font=("Segoe UI", 14),
                                  fg=self.colors["text"],
                                  bg=self.colors["light"])
        self.timer_label.pack(side="left")

        # Game board container with subtle shadow effect
        board_container = tk.Frame(game_container, bg=self.colors["background"], 
                                 padx=10, pady=10)
        board_container.pack(expand=True)
        
        # Game grid
        self.frames = []
        self.buttons = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
        self.create_grid(board_container)

        # Game controls with modern styling
        controls_frame = tk.Frame(game_container, bg=self.colors["background"])
        controls_frame.pack(fill="x", pady=(15, 0))
        
        # Stats display
        self.stats_frame = tk.Frame(controls_frame, bg=self.colors["light"], bd=1, relief="solid")
        self.stats_frame.pack(side="top", fill="x", pady=(0, 10))
        
        stats_inner = tk.Frame(self.stats_frame, bg=self.colors["light"], padx=15, pady=8)
        stats_inner.pack(fill="x")
        
        # Player X stats
        x_stats = tk.Frame(stats_inner, bg=self.colors["light"])
        x_stats.pack(side="left", expand=True)
        
        x_label = tk.Label(x_stats, text="Player X", font=("Segoe UI", 10, "bold"),
                          fg=self.colors["x_color"], bg=self.colors["light"])
        x_label.pack()
        
        self.x_wins_label = tk.Label(x_stats, text="Wins: 0", font=("Segoe UI", 10),
                                   fg=self.colors["text"], bg=self.colors["light"])
        self.x_wins_label.pack()
        
        # Draw stats
        draw_stats = tk.Frame(stats_inner, bg=self.colors["light"])
        draw_stats.pack(side="left", expand=True)
        
        draw_label = tk.Label(draw_stats, text="Draws", font=("Segoe UI", 10, "bold"),
                            fg=self.colors["dark"], bg=self.colors["light"])
        draw_label.pack()
        
        self.draws_label = tk.Label(draw_stats, text="Total: 0", font=("Segoe UI", 10),
                                  fg=self.colors["text"], bg=self.colors["light"])
        self.draws_label.pack()
        
        # Player O stats
        o_stats = tk.Frame(stats_inner, bg=self.colors["light"])
        o_stats.pack(side="left", expand=True)
        
        o_label = tk.Label(o_stats, text="Player O", font=("Segoe UI", 10, "bold"),
                          fg=self.colors["o_color"], bg=self.colors["light"])
        o_label.pack()
        
        self.o_wins_label = tk.Label(o_stats, text="Wins: 0", font=("Segoe UI", 10),
                                   fg=self.colors["text"], bg=self.colors["light"])
        self.o_wins_label.pack()
        
        # Button container
        button_container = tk.Frame(controls_frame, bg=self.colors["background"])
        button_container.pack(fill="x")

        # Reset button with icon
        reset_button = tk.Button(button_container, text="New Game", 
                               font=("Segoe UI", 12, "bold"),
                               bg=self.colors["secondary"], fg="white",
                               activebackground=self.colors["secondary"],
                               activeforeground="white",
                               relief="flat", bd=0,
                               padx=15, pady=8,
                               command=self.reset_game)
        reset_button.pack(side="left", padx=(0, 5), expand=True, fill="x")

        # Back button with icon
        back_button = tk.Button(button_container, text="Back to Menu", 
                              font=("Segoe UI", 12, "bold"),
                              bg=self.colors["dark"], fg="white",
                              activebackground=self.colors["dark"],
                              activeforeground="white",
                              relief="flat", bd=0,
                              padx=15, pady=8,
                              command=self.back_to_menu)
        back_button.pack(side="right", padx=(5, 0), expand=True, fill="x")

    def create_grid(self, parent):
        # Create a container for all layers
        layers_container = tk.Frame(parent, bg=self.colors["background"])
        layers_container.pack()
        
        # Layer colors - subtle gradient for depth perception
        layer_colors = [
            self.colors["light"],  # Layer 1 (lightest)
            self.adjust_color_brightness(self.colors["light"], 0.97),  # Layer 2
            self.adjust_color_brightness(self.colors["light"], 0.94),  # Layer 3
            self.adjust_color_brightness(self.colors["light"], 0.91),  # Layer 4 (darkest)
        ]
        
        for z in range(4):
            # Create a frame with a title bar for each layer
            layer_frame = tk.Frame(layers_container, bg=self.colors["background"])
            layer_frame.grid(row=0, column=z, padx=8, pady=5)
            
            # Title bar with layer number
            title_bar = tk.Frame(layer_frame, bg=self.colors["primary"], padx=5, pady=3)
            title_bar.pack(fill="x")
            
            layer_label = tk.Label(title_bar, text=f"Layer {z+1}", 
                                 font=("Segoe UI", 12, "bold"),
                                 fg="white", bg=self.colors["primary"])
            layer_label.pack()
            
            # Grid frame with border
            grid_frame = tk.Frame(layer_frame, bg=self.colors["dark"], bd=1, relief="solid")
            grid_frame.pack()
            
            # Inner frame with background color
            inner_frame = tk.Frame(grid_frame, bg=layer_colors[z], padx=4, pady=4)
            inner_frame.pack()
            
            self.frames.append(inner_frame)
            
            # Create the grid of buttons
            for x in range(4):
                for y in range(4):
                    # Button with modern styling
                    button = tk.Button(
                        inner_frame, 
                        text="", 
                        width=3, 
                        height=1,  # Smaller height for better proportions
                        font=("Segoe UI", 14, "bold"),
                        bg="white",
                        activebackground=self.colors["light"],
                        relief="flat",
                        bd=1,
                        padx=2, 
                        pady=2,
                        command=lambda x=x, y=y, z=z: self.make_move(x, y, z)
                    )
                    button.grid(row=x, column=y, padx=2, pady=2, sticky="nsew")
                    self.buttons[z][x][y] = button
                    
                    # Add hover effect
                    button.bind("<Enter>", lambda e, btn=button: self.on_button_hover(btn, True))
                    button.bind("<Leave>", lambda e, btn=button: self.on_button_hover(btn, False))
    
    def adjust_color_brightness(self, hex_color, factor):
        """Adjust the brightness of a hex color"""
        # Convert hex to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        # Convert to HSV, adjust brightness (V), convert back to RGB
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        v = max(0, min(1, v * factor))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert back to hex
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    
    def on_button_hover(self, button, is_hover):
        """Add hover effect to buttons"""
        if button["state"] != "disabled":
            if is_hover:
                button.config(bg=self.colors["light"])
            else:
                button.config(bg="white")

    def show_menu(self):
        self.game_frame.pack_forget()
        self.menu_frame.pack(fill="both", expand=True)

    def toggle_ai_options(self):
        # Show or hide AI algorithm options based on game mode
        if self.mode_var.get() == "ai":
            self.ai_frame.pack(pady=10)
        else:
            self.ai_frame.pack_forget()
            
    def start_game(self):
        # Check if game mode has changed
        new_game_mode = self.mode_var.get()
        mode_changed = (new_game_mode != self.game_mode)
        
        # Update game mode based on selection
        self.game_mode = new_game_mode
        
        # Reset statistics if game mode has changed
        if mode_changed:
            self.stats = {
                "x_wins": 0,
                "o_wins": 0,
                "draws": 0,
                "games_played": 0
            }
            self.update_stats()
            print(f"Game mode changed to {self.game_mode}. Statistics reset.")
        
        # Update AI algorithm if playing against AI
        if self.game_mode == "ai":
            # Get the selected algorithm code
            selected_name = self.algorithm_var.get()
            selected_code = "heuristic"  # Default
            
            # Find the matching algorithm code
            for code, name in self.ai_algorithms:
                if name == selected_name:
                    selected_code = code
                    break
            
            # Update the AI's algorithm
            self.ai.set_algorithm(selected_code)
            print(f"AI using algorithm: {self.ai.get_algorithm_name()}")
        
        self.menu_frame.pack_forget()
        self.game_frame.pack(fill="both", expand=True)
        self.reset_game()
        self.start_timer()

    def reset_game(self):
        # Reset game state
        self.game.reset_game()
        
        # Reset UI elements
        self.update_grid()
        self.update_player_indicator()
        
        # Update status label based on game mode with styling
        if self.game_mode == "ai":
            ai_name = self.ai.get_algorithm_name()
            self.status_label.config(
                text=f"Player X's Turn", 
                fg=self.colors["x_color"]
            )
        else:
            self.status_label.config(
                text=f"Player {self.game.current_player}'s Turn",
                fg=self.colors["x_color"]
            )
        
        # Reset and start timer    
        self.start_time = time.time()
        if not self.timer_running:
            self.timer_running = True
            self.update_timer()

        # Reset all buttons
        for z in range(4):
            for x in range(4):
                for y in range(4):
                    self.buttons[z][x][y].config(
                        state="normal", 
                        text="", 
                        fg=self.colors["text"], 
                        bg="white",
                        relief="flat"
                    )
        
        # Add subtle animation for reset
        self.animate_reset()
    
    def animate_reset(self):
        """Add a subtle animation when resetting the game"""
        # Flash the board briefly
        for z in range(4):
            for x in range(4):
                for y in range(4):
                    self.buttons[z][x][y].config(bg=self.colors["light"])
        
        self.root.update()
        self.root.after(100, lambda: self.update_grid())

    def back_to_menu(self):
        # Create a modern confirmation dialog
        confirm_frame = tk.Frame(self.game_frame, bg=self.colors["light"], 
                               bd=2, relief="solid", padx=25, pady=20)
        confirm_frame.place(relx=0.5, rely=0.4, anchor="center")
        
        confirm_label = tk.Label(confirm_frame, 
                                text="Return to main menu?", 
                                font=("Segoe UI", 14, "bold"),
                                fg=self.colors["dark"], 
                                bg=self.colors["light"])
        confirm_label.pack(pady=(0, 15))
        
        # Button frame for yes/no
        button_frame = tk.Frame(confirm_frame, bg=self.colors["light"])
        button_frame.pack()
        
        # Yes button
        yes_button = tk.Button(button_frame, text="Yes", 
                             font=("Segoe UI", 12),
                             bg=self.colors["primary"], fg="white",
                             activebackground=self.colors["primary"],
                             activeforeground="white",
                             relief="flat", bd=0,
                             padx=15, pady=5,
                             command=lambda: self.confirm_back_to_menu(confirm_frame))
        yes_button.pack(side="left", padx=(0, 10))
        
        # No button
        no_button = tk.Button(button_frame, text="No", 
                            font=("Segoe UI", 12),
                            bg=self.colors["dark"], fg="white",
                            activebackground=self.colors["dark"],
                            activeforeground="white",
                            relief="flat", bd=0,
                            padx=15, pady=5,
                            command=lambda: confirm_frame.destroy())
        no_button.pack(side="right")
    
    def confirm_back_to_menu(self, dialog_frame):
        """Handle confirmation to return to menu"""
        dialog_frame.destroy()
        self.timer_running = False
        self.show_menu()

    def on_close(self):
        # Create a modern confirmation dialog
        confirm_frame = tk.Frame(self.root, bg=self.colors["light"], 
                               bd=2, relief="solid", padx=25, pady=20)
        confirm_frame.place(relx=0.5, rely=0.4, anchor="center")
        
        confirm_label = tk.Label(confirm_frame, 
                                text="Are you sure you want to quit?", 
                                font=("Segoe UI", 14, "bold"),
                                fg=self.colors["dark"], 
                                bg=self.colors["light"])
        confirm_label.pack(pady=(0, 15))
        
        # Button frame for yes/no
        button_frame = tk.Frame(confirm_frame, bg=self.colors["light"])
        button_frame.pack()
        
        # Yes button
        yes_button = tk.Button(button_frame, text="Yes", 
                             font=("Segoe UI", 12),
                             bg=self.colors["primary"], fg="white",
                             activebackground=self.colors["primary"],
                             activeforeground="white",
                             relief="flat", bd=0,
                             padx=15, pady=5,
                             command=self.root.destroy)
        yes_button.pack(side="left", padx=(0, 10))
        
        # No button
        no_button = tk.Button(button_frame, text="No", 
                            font=("Segoe UI", 12),
                            bg=self.colors["dark"], fg="white",
                            activebackground=self.colors["dark"],
                            activeforeground="white",
                            relief="flat", bd=0,
                            padx=15, pady=5,
                            command=lambda: confirm_frame.destroy())
        no_button.pack(side="right")

    def make_move(self, x, y, z):
        if self.game.board[x][y][z] is None and self.timer_running:
            # Visual feedback for the click
            self.buttons[z][x][y].config(relief="sunken")
            self.root.update()
            self.root.after(100, lambda: self.buttons[z][x][y].config(relief="flat"))
            
            # Make the move
            self.game.board[x][y][z] = self.game.current_player
            self.update_grid()
            
            # Check for win
            if self.game.is_winner(self.game.current_player):
                winner = self.game.current_player
                result = "X_WIN" if winner == "X" else "O_WIN"
                
                # Update status and visuals
                self.status_label.config(
                    text=f"Player {winner} Wins!",
                    fg=self.colors["x_color"] if winner == "X" else self.colors["o_color"]
                )
                self.highlight_winner(self.game.winning_positions)
                self.disable_buttons()
                self.timer_running = False
                
                # Show result popup
                self.root.after(1000, lambda: self.show_game_result(result))
                return
                
            # Check for draw
            elif self.game.is_full():
                self.status_label.config(text="It's a Draw", fg=self.colors["dark"])
                self.disable_buttons()
                self.timer_running = False
                
                # Show draw result popup
                self.root.after(1000, lambda: self.show_game_result("DRAW"))
                return
                
            # Switch players
            next_player = "O" if self.game.current_player == "X" else "X"
            self.game.current_player = next_player
            self.update_player_indicator()
            
            # If playing against AI and it's AI's turn
            if self.game_mode == "ai" and self.game.current_player == "O":
                ai_name = self.ai.get_algorithm_name()
                self.status_label.config(
                    text=f"Player O's Turn (AI)",
                    fg=self.colors["o_color"]
                )
                self.root.after(500, self.ai_move)
            else:
                # Human player's turn
                self.status_label.config(
                    text=f"Player {self.game.current_player}'s Turn",
                    fg=self.colors["x_color"] if self.game.current_player == "X" else self.colors["o_color"]
                )

    def ai_move(self):
        if not self.timer_running:
            return
            
        # Show thinking indicator
        thinking_label = tk.Label(self.game_frame, text="AI is thinking...", 
                                font=("Segoe UI", 12, "italic"),
                                fg=self.colors["o_color"],
                                bg=self.colors["background"])
        thinking_label.place(relx=0.5, rely=0.1, anchor="center")
        self.root.update()
        
        # Get AI move with timing
        start_time = time.time()
        move = self.ai.find_best_move(self.game)
        end_time = time.time()
        time_taken = end_time - start_time

        # Remove thinking indicator
        thinking_label.destroy()
        
        # Log AI performance
        print(f"AI algorithm: {self.ai.get_algorithm_name()}")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        if move:
            x, y, z = move
            
            # Highlight the AI's move
            self.buttons[z][x][y].config(bg=self.colors["light"])
            self.root.update()
            self.root.after(300, lambda: self.buttons[z][x][y].config(bg="white"))
            self.root.after(300, lambda: self.buttons[z][x][y].config(bg=self.colors["light"]))
            self.root.after(600, lambda: self.buttons[z][x][y].config(bg="white"))
            
            # Make the move
            self.game.board[x][y][z] = self.game.current_player
            self.update_grid()
            
            # Check for win
            if self.game.is_winner(self.game.current_player):
                winner = self.game.current_player
                result = "O_WIN"
                
                # Update status and visuals
                self.status_label.config(
                    text=f"Player {winner} Wins!",
                    fg=self.colors["o_color"]
                )
                self.highlight_winner(self.game.winning_positions)
                self.disable_buttons()
                self.timer_running = False
                
                # Show result popup
                self.root.after(1000, lambda: self.show_game_result(result))
                return
                
            # Check for draw
            elif self.game.is_full():
                self.status_label.config(text="It's a Draw", fg=self.colors["dark"])
                self.disable_buttons()
                self.timer_running = False
                
                # Show draw result popup
                self.root.after(1000, lambda: self.show_game_result("DRAW"))
                return
                
        # Switch back to human player
        self.game.current_player = "X"
        self.update_player_indicator()
        self.status_label.config(text="Player X's Turn", fg=self.colors["x_color"])

    def update_grid(self):
        for z in range(4):
            for x in range(4):
                for y in range(4):
                    symbol = self.game.board[x][y][z]
                    if symbol == "X":
                        self.buttons[z][x][y].config(
                            text=symbol, 
                            fg=self.colors["x_color"],
                            font=("Segoe UI", 14, "bold"),
                            bg="white"
                        )
                    elif symbol == "O":
                        self.buttons[z][x][y].config(
                            text=symbol, 
                            fg=self.colors["o_color"],
                            font=("Segoe UI", 14, "bold"),
                            bg="white"
                        )
                    else:
                        self.buttons[z][x][y].config(
                            text="", 
                            fg=self.colors["text"],
                            bg="white"
                        )

    def highlight_winner(self, winning_positions):
        # First flash the winning line
        self.flash_winning_line(winning_positions, 0)
        
        # Then set the final highlight
        for (x, y, z) in winning_positions:
            self.buttons[z][x][y].config(
                bg=self.colors["highlight"],
                relief="raised"
            )
    
    def flash_winning_line(self, positions, count):
        """Create a flashing effect for the winning line"""
        if count >= 6:  # Flash 3 times (on/off cycles)
            return
            
        # Toggle between highlight and white
        highlight = count % 2 == 0
        
        for (x, y, z) in positions:
            if highlight:
                self.buttons[z][x][y].config(bg=self.colors["highlight"])
            else:
                self.buttons[z][x][y].config(bg="white")
                
        # Schedule the next flash
        self.root.after(200, lambda: self.flash_winning_line(positions, count + 1))

    def disable_buttons(self):
        for z in range(4):
            for x in range(4):
                for y in range(4):
                    if self.game.board[x][y][z] is None:
                        # Only disable empty cells with a subtle gray color
                        self.buttons[z][x][y].config(
                            state="disabled",
                            bg="#f0f0f0"  # Light gray for disabled
                        )
                    else:
                        # Keep the played cells as is, but disable them
                        self.buttons[z][x][y].config(state="disabled")

    def start_timer(self):
        self.start_time = time.time()
        if not self.timer_running:
            self.timer_running = True
            self.update_timer()

    def update_timer(self):
        if not self.timer_running:
            return
        elapsed = int(time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self.update_timer)
        
    def update_player_indicator(self):
        """Update the player indicator based on current player"""
        if self.game.current_player == "X":
            self.player_indicator.delete("all")
            self.player_indicator.create_oval(2, 2, 18, 18, fill=self.colors["x_color"], outline="")
        else:
            self.player_indicator.delete("all")
            self.player_indicator.create_oval(2, 2, 18, 18, fill=self.colors["o_color"], outline="")
    
    def update_stats(self):
        """Update the game statistics display"""
        self.x_wins_label.config(text=f"Wins: {self.stats['x_wins']}")
        self.o_wins_label.config(text=f"Wins: {self.stats['o_wins']}")
        self.draws_label.config(text=f"Total: {self.stats['draws']}")
        
    def show_game_result(self, result):
        """Display game result with animation"""
        if result == "X_WIN":
            self.stats["x_wins"] += 1
            self.stats["games_played"] += 1
            result_text = "Player X Wins!"
            result_color = self.colors["x_color"]
        elif result == "O_WIN":
            self.stats["o_wins"] += 1
            self.stats["games_played"] += 1
            result_text = "Player O Wins!"
            result_color = self.colors["o_color"]
        else:  # Draw
            self.stats["draws"] += 1
            self.stats["games_played"] += 1
            result_text = "It's a Draw!"
            result_color = self.colors["dark"]
            
        # Update stats display
        self.update_stats()
        
        # Create a result popup
        result_frame = tk.Frame(self.game_frame, bg=self.colors["light"], 
                              bd=2, relief="solid", padx=20, pady=15)
        result_frame.place(relx=0.5, rely=0.4, anchor="center")
        
        result_label = tk.Label(result_frame, text=result_text, 
                               font=("Segoe UI", 18, "bold"),
                               fg=result_color, bg=self.colors["light"])
        result_label.pack(pady=(0, 10))
        
        # Add a continue button
        continue_button = tk.Button(result_frame, text="Continue", 
                                  font=("Segoe UI", 12, "bold"),
                                  bg=self.colors["primary"], fg="white",
                                  activebackground=self.colors["primary"],
                                  activeforeground="white",
                                  relief="flat", bd=0,
                                  padx=15, pady=5,
                                  command=lambda: result_frame.destroy())
        continue_button.pack()
