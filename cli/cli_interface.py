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
            substage: Current substage ("診斷", "對話", or "總結")
            question_idx: Current question index (for "對話" substage)
            total_questions: Total number of questions (for "對話" substage)
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
    
    def get_user_input(self, prompt: Optional[str] = None) -> str:
        """
        Get multi-line user input.
        
        Args:
            prompt: Optional system prompt/question to display first
            
        Returns:
            User's input as a single string (multi-line preserved)
        """
        # If there's a prompt, show it as a system message
        if prompt:
            self.show_message("system", prompt)
        
        # Display input instructions
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
        self._refresh_display()


# Example usage
if __name__ == "__main__":
    # Create CLI interface
    cli = CLIInterface()
    
    # Simulate stage 1, diagnosis
    cli.update_stage(stage_idx=1, substage="診斷")
    
    # Simulate stage 1, question phase
    cli.update_stage(stage_idx=1, substage="對話", question_idx=1, total_questions=3)
    
    # Get user input
    answer1 = cli.get_user_input("你希望老師是什麼風格？")
    
    # Simulate follow-up question
    answer2 = cli.get_user_input("能具體說明「輕鬆有趣」是什麼意思嗎？")
    
    # Move to next question (clear conversation)
    cli.clear_conversation()
    cli.update_stage(stage_idx=1, substage="對話", question_idx=2, total_questions=3)
    
    answer3 = cli.get_user_input("你希望用什麼教學方法？")
    
    # Move to integration phase
    cli.update_stage(stage_idx=1, substage="總結")
    cli.show_message("system", "正在整合你的答案，生成優化後的提示詞...")
    
    print("\n測試完成！")