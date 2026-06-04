import os
import sys
from dotenv import load_dotenv

# Try importing groq to prevent unbound linting issues later
try:
    # pyrefly: ignore [missing-import]
    from groq import Groq
except ImportError:
    print("\033[91m\033[1mError:\033[0m The 'groq' package is not installed.")
    print("Please activate your virtual environment and run: pip install -r requirements.txt")
    sys.exit(1)

# ANSI Color Codes for premium terminal aesthetics
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"

def main():
    # Force enabling ANSI colors on Windows systems if supported
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    # Load environment variables from .env file
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY")

    # Interactive key setup if missing or placeholder
    if not groq_key or groq_key.startswith("your_"):
        print(f"\n{YELLOW}⚠️ No active GROQ_API_KEY found in your '.env' file.{RESET}")
        print(f"{GRAY}If you do not have an API key, you can get one for free at: {CYAN}https://console.groq.com/{RESET}")
        try:
            # Prompt user to input the key in the terminal
            input_key = input(f"\n{BOLD}{CYAN}🔑 Please paste your Groq API Key (starts with gsk_): {RESET}").strip()
            if not input_key:
                print(f"\n{RED}{BOLD}Error:{RESET} No API key entered. Exiting application.")
                sys.exit(1)
            
            # Save the key into the .env file automatically
            with open(".env", "w", encoding="utf-8") as env_file:
                env_file.write("# Real-Time Streaming Chatbot Environment Configuration\n\n")
                env_file.write("# Groq API Key (Ultra-fast LPU inference)\n")
                env_file.write("# Get your key at: https://console.groq.com/\n")
                env_file.write(f"GROQ_API_KEY={input_key}\n")
            
            groq_key = input_key
            print(f"\n{GREEN}{BOLD}✓ API Key successfully saved to '.env'!{RESET}")
            
            # Reload environment variables
            load_dotenv()
        except KeyboardInterrupt:
            print(f"\n\n{RED}Setup cancelled. Exiting.{RESET}")
            sys.exit(1)

    # Initialize client and select model
    client = Groq(api_key=groq_key)
    model_name = "llama-3.3-70b-versatile"

    # Initialize message list for conversation context retention
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful, direct, and intelligent AI assistant powered by Groq's ultra-fast LPU inference engine."
        }
    ]

    # Display welcome message with premium look
    print(f"\n{CYAN}===================================================================={RESET}")
    print(f"{BOLD}{GREEN}  🚀 Real-Time Streaming Chatbot Initialized Successfully{RESET}")
    print(f"  {BOLD}Provider:{RESET} Groq")
    print(f"  {BOLD}Model:{RESET}    {model_name}")
    print(f"  {GRAY}Type 'exit' or 'quit' to end the session. Ctrl+C to force quit.{RESET}")
    print(f"{CYAN}===================================================================={RESET}")

    while True:
        try:
            # Display colored prompt for user input
            user_input = input(f"\n{BOLD}{BLUE}You:{RESET} ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print(f"\n{YELLOW}Exiting chatbot session. Goodbye! 👋{RESET}\n")
                break

            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Print assistant label with trailing space, prepared for stream chunk reception
            print(f"\n{BOLD}{GREEN}Assistant (Groq):{RESET} ", end="", flush=True)

            full_response = ""
            
            # Call completions API with streaming enabled
            stream = client.chat.completions.create(
                messages=messages,
                model=model_name,
                stream=True,
                temperature=0.7
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    # Print chunk immediately without newline and flush output stream
                    print(delta, end="", flush=True)
                    full_response += delta
            
            # Print a final newline after streaming completes
            print()
            
            # Save assistant's completed response to conversation history
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Session interrupted by user (Ctrl+C). Goodbye! 👋{RESET}\n")
            break
        except Exception as e:
            error_message = str(e)
            print(f"\n\n{RED}{BOLD}API Error:{RESET} {error_message}")
            
            # Contextual suggestions for common failures
            if "api_key" in error_message.lower() or "auth" in error_message.lower() or "unauthorized" in error_message.lower():
                print(f"{YELLOW}Hint: Please check if your API key in the '.env' file is valid, spelled correctly, and not revoked.{RESET}")
            elif "rate_limit" in error_message.lower() or "429" in error_message or "limit" in error_message.lower():
                print(f"{YELLOW}Hint: You've hit a rate limit. Wait a moment or review your account limits.{RESET}")
            elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                print(f"{YELLOW}Hint: Network error occurred. Please verify your internet connection.{RESET}")
            else:
                print(f"{YELLOW}Hint: Ensure your dependencies are fully installed and you are using compatible models.{RESET}")

if __name__ == "__main__":
    main()
