import os
from typing import Optional, List


class CLIInterface:
    """
    Command-line interface for user interaction with the agent system.
    Provides methods for displaying stage information, messages, and collecting user input.
    """
    
    def __init__(self, terminal_width: int = 80):
        """
        Initialize CLI interface.
        
        Args:
            terminal_width: Width of the terminal display (default: 80)
        """
        self.terminal_width = terminal_width
        self.conversation_buffer: List[tuple[str, str]] = []  # [(role, message), ...]
        self.max_buffer_size = 10
        self.pending_system_message_index: Optional[int] = None
        self.waiting_message_text = "等待LLM回覆中"
        
        # Current stage information
        self.current_stage = 1
        self.current_substage = ""
        self.current_question_idx = None
        self.total_questions = None
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _render_header(self):
        """Render the stage information header"""
        separator = "=" * self.terminal_width
        
        # Build stage info string
        stage_info = f"階段 {self.current_stage}/6"
        
        if self.current_substage:
            if self.current_substage == "對話" and self.current_question_idx is not None:
                stage_info += f" - {self.current_substage} ({self.current_question_idx}/{self.total_questions})"
            else:
                stage_info += f" - {self.current_substage}"
        
        print(separator)
        print(stage_info.center(self.terminal_width))
        print(separator)
        print()
    
    def _render_message(self, role: str, message: str):
        """
        Render a single message with left/right alignment.
        
        Args:
            role: "system" or "user"
            message: The message content
        """
        if role == "system":
            # System message: left-aligned
            print(f"系統: {message}")
        else:
            # User message: right-aligned
            prefix = "你: "
            max_content_width = self.terminal_width - len(prefix) - 2
            
            # Handle multi-line messages
            lines = message.split('\n')
            for line in lines:
                # Calculate padding for right alignment
                padding = self.terminal_width - len(prefix) - len(line)
                if padding > 0:
                    print(" " * padding + prefix + line)
                else:
                    # If message is too long, truncate
                    print(" " * 10 + prefix + line[:max_content_width])
        
        print()  # Add spacing between messages
    
    def _render_conversation(self):
        """Render all messages in the conversation buffer"""
        for role, message in self.conversation_buffer:
            self._render_message(role, message)
    
    def _refresh_display(self):
        """Refresh the entire display (clear and redraw)"""
        self._clear_screen()
        self._render_header()
        self._render_conversation()
    
    def update_stage(self, stage_idx: int, substage: str,
                    question_idx: Optional[int] = None,
                    total_questions: Optional[int] = None):
        """
        Update stage information and refresh display.
        
        Args:
            stage_idx: Current stage number (1-6)
            substage: Current substage label (e.g., "diagnosis", "dialogue", "integration")
            question_idx: Current question index (used for the dialogue substage)
            total_questions: Total number of questions (used for the dialogue substage)
        """
        self.current_stage = stage_idx
        self.current_substage = substage
        self.current_question_idx = question_idx
        self.total_questions = total_questions
        
        self._refresh_display()
    
    def show_message(self, role: str, message: str):
        """
        Display a message and add it to conversation buffer.
        
        Args:
            role: "system" or "user"
            message: The message content
        """
        # Add to buffer
        self.conversation_buffer.append((role, message))
        
        # Maintain buffer size limit
        if len(self.conversation_buffer) > self.max_buffer_size:
            self.conversation_buffer.pop(0)
        
        # Refresh display
        self._refresh_display()

    def _replace_pending_system_message(self, message: str) -> bool:
        """Replace a pending system message if one exists."""
        idx = self.pending_system_message_index
        if idx is None:
            return False
        if idx < 0 or idx >= len(self.conversation_buffer):
            self.pending_system_message_index = None
            return False
        self.conversation_buffer[idx] = ("system", message)
        self.pending_system_message_index = None
        self._refresh_display()
        return True

    def show_system_message(self, message: str):
        """Display a system message, replacing a pending waiting message if present."""
        if not self._replace_pending_system_message(message):
            self.show_message("system", message)

    def show_waiting_message(self, message: Optional[str] = None):
        """Display a waiting placeholder for an upcoming system response."""
        if message is None:
            message = self.waiting_message_text
        idx = self.pending_system_message_index
        if idx is not None and 0 <= idx < len(self.conversation_buffer):
            self.conversation_buffer[idx] = ("system", message)
            self._refresh_display()
            return
        self.show_message("system", message)
        self.pending_system_message_index = len(self.conversation_buffer) - 1

    def clear_waiting_message(self):
        """Remove a pending waiting placeholder if it exists."""
        idx = self.pending_system_message_index
        if idx is None:
            return
        if 0 <= idx < len(self.conversation_buffer):
            self.conversation_buffer.pop(idx)
        self.pending_system_message_index = None
        self._refresh_display()
    
    def get_user_input(self, prompt: Optional[str] = None, options: Optional[List[str]] = None) -> str:
        """
        Get multi-line user input, optionally with multiple choice options.
        
        Args:
            prompt: Optional system prompt/question to display first
            options: Optional list of options (e.g., ["A) ...", "B) ...", "C) Other", "D) None"])
        
        Returns:
            User's input as a single string (multi-line preserved)
        """
        # Build the full prompt message
        full_message = ""
        
        if prompt:
            full_message = prompt
        
        # If options provided, format them nicely
        if options:
            full_message += "\n\n"
            for option in options:
                full_message += f"{option}\n"
        
        # Show the prompt as a system message
        if full_message:
            self.show_system_message(full_message.strip())
        
        # Display input instructions
        if options:
            print("請輸入你的選擇（或自由回答，空行結束）:")
        else:
            print("請輸入你的回答（空行結束）:")
        
        # Collect multi-line input
        lines = []
        while True:
            try:
                line = input("> ")
                if line.strip() == "":
                    break
                lines.append(line)
            except EOFError:
                break
        
        user_input = "\n".join(lines)
        
        # Show user's message in the conversation
        self.show_message("user", user_input)
        
        return user_input
    
    def clear_conversation(self):
        """Clear the conversation buffer (called when moving to next question)"""
        self.conversation_buffer.clear()
        self.pending_system_message_index = None
        self._refresh_display()


# Example usage
if __name__ == "__main__":
    # Create CLI interface
    cli = CLIInterface()
    
    # Simulate stage 1, diagnosis
    cli.update_stage(stage_idx=1, substage="診斷")
    
    # Simulate stage 1, question phase - open-ended question
    cli.update_stage(stage_idx=1, substage="對話", question_idx=1, total_questions=3)
    answer1 = cli.get_user_input("你希望老師是什麼風格？")
    
    # Simulate follow-up with options
    options = [
        "A) 逐步講解，從基礎開始",
        "B) 實作導向，透過專案學習",
        "C) 理論與實作並重",
        "D) 其他",
        "E) 沒有想法"
    ]
    answer2 = cli.get_user_input("請選擇你偏好的教學方式：", options=options)
    
    # Move to next question (clear conversation)
    cli.clear_conversation()
    cli.update_stage(stage_idx=1, substage="對話", question_idx=2, total_questions=3)
    
    answer3 = cli.get_user_input("你希望用什麼教學方法？")
    
    # Move to integration phase
    cli.update_stage(stage_idx=1, substage="總結")
    cli.show_message("system", "正在整合你的答案，生成優化後的提示詞...")
    
    print("\n測試完成！")
