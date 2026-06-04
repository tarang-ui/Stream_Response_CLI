# Real-Time Streaming Responses from Groq API Using Python

This project is a terminal-based interactive chatbot built with Python. It demonstrates how to achieve real-time streaming of large language model (LLM) responses using the **Groq API** (LPU-accelerated Inference Engine). 

By outputting text progressively as it is generated, this application minimizes perceived latency and delivers a premium, highly responsive user experience.

---

## 🏗️ Architecture Flow

The interaction logic follows a linear pathway from input acquisition to real-time rendering:

```
[User Input]
     │
     ▼
[Prompt & Context Construction] (Appending to history list)
     │
     ▼
[API Client Request] (Payload containing stream=True)
     │
     ▼
[LLM Streaming Engine] (Iteratively generates tokens)
     │
     ▼
[HTTP Chunk Reception] (Receiving Server-Sent Events)
     │
     ▼
[Terminal Output Buffer Flush] (sys.stdout.write / print end="" with flush=True)
     │
     ▼
[Append to Conversation History] (Loop repeats)
```

---

## ⚡ Why Streaming? Buffering vs. Streaming UX

Traditional API calls for LLM completions use **buffering**. The system waits for the model to generate the entire response on the server, packages it into a single JSON object, and sends it back over a single HTTP connection.

### ⏱️ Time to First Token (TTFT)
* **Buffering**: If a model generates a 500-word response at 50 tokens per second, the server will take roughly **10 seconds** to generate the text. The user sees a freezing spinner or blank screen for the entire 10 seconds before the text suddenly pops up.
* **Streaming**: The server sends each token (or word fragment) as soon as it is computed. The Time to First Token (TTFT) is reduced to a fraction of a second (often **50ms - 200ms**). Even though the overall completion still takes 10 seconds, the user is actively reading the response, making the system feel instantaneous.

### 📡 Server-Sent Events (SSE)
Under the hood, setting `stream=True` initiates an HTTP connection using `Transfer-Encoding: chunked`. The API server responds using **Server-Sent Events (SSE)**, sending data packets prefixed with `data:`. The Python client reads this persistent stream, parses each JSON data chunk, and yields it to our execution loop in real-time.

---

## 🛠️ Step-by-Step Setup Instructions

Follow these instructions to run the chatbot locally on your Windows machine:

### 1. Initialize Virtual Environment
Using a virtual environment isolates project-specific dependencies and prevents conflicts with other system-wide packages.

Open **PowerShell** or **Command Prompt** in the project directory (`d:\llm-streaming-demo`) and run:
```powershell
# Create the virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Or on Windows Command Prompt:
.\venv\Scripts\activate.bat
```

### 2. Install Project Dependencies
With the virtual environment active, install the required packages:
```powershell
pip install -r requirements.txt
```
This installs:
* `groq`: The official Python SDK for Groq.
* `python-dotenv`: A tool to load key-value pairs from a `.env` file into system environment variables.

### 3. Configure API Credentials
For security, do not hardcode your API keys. Instead, use environment variables:

1. Copy the template configuration file:
   ```powershell
   copy .env.template .env
   ```
2. Open the newly created `.env` file in your text editor.
3. Insert your API key:
   * **For Groq**: Set `GROQ_API_KEY=gsk_your_key_here` (get yours at [console.groq.com](https://console.groq.com/)).

---

## 🔍 Detailed Code Walkthrough

The script `chatbot.py` contains several critical components that implement the requirements:

### 1. Configuration Loading
```python
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
```
Loads `.env` variables. It verifies if `GROQ_API_KEY` is present and raises a configuration error if it is missing or unchanged.

### 2. Conversation Memory Structure
```python
messages = []
messages.append({"role": "system", "content": "..."})
```
An LLM is stateless. To maintain context, we append the entire conversation history to the `messages` list. Every prompt submission includes the history so the assistant remembers what was said earlier.

### 3. The Core Streaming Loop
```python
stream = client.chat.completions.create(
    messages=messages,
    model=model_name,
    stream=True,  # Critical parameter
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta is not None:
        print(delta, end="", flush=True)
        full_response += delta
```
* `stream=True`: Orders the server to stream response tokens.
* `for chunk in stream`: Suspends execution until the next packet arrives, then resumes.
* `end=""`: Prevents Python from appending a newline after printing each individual token chunk.
* `flush=True`: Bypasses terminal buffer latency, forcing the text to render on screen immediately.

---

## 🛡️ Error Handling & Troubleshooting

Our implementation wraps the streaming request in a `try-except` block to capture errors cleanly:

| Error Type | Common Cause | Mitigation / Fix |
| :--- | :--- | :--- |
| **Authentication Error** (`401`) | Missing, expired, or mistyped API key in `.env`. | Verify your key on the Groq console. Ensure the `.env` filename is exactly `.env` (not `.env.txt`). |
| **Rate Limit Error** (`429`) | Exceeded token-per-minute (TPM) or requests-per-minute (RPM) limits. | Implement backoff logic, wait, or upgrade your API tier. |
| **Network Timeout** | Interrupted internet connection or API server outage. | Check your local network. Test again after checking status pages ([status.groq.com](https://status.groq.com)). |
| **KeyboardInterrupt** (`Ctrl+C`) | User forcibly terminating execution. | Caught by `except KeyboardInterrupt` to exit gracefully without tracebacks. |

---

## 🧪 Testing Approach & Sample Execution

To run your chatbot:
```powershell
python chatbot.py
```

### Sample Session Output

```text
====================================================================
  🚀 Real-Time Streaming Chatbot Initialized Successfully
  Provider: Groq
  Model:    llama-3.3-70b-versatile
  Type 'exit' or 'quit' to end the session. Ctrl+C to force quit.
====================================================================

You: hello, tell me a 1-sentence joke.

Assistant (Groq): Why don't scientists trust atoms? Because they make up everything!

You: explain that joke.

Assistant (Groq): The joke relies on a pun: "make up" refers both to scientists discovering that atoms physically compose all matter, and the idiom for fabricating lies or exaggerations.

You: quit

Exiting chatbot session. Goodbye! 👋
```

---

## 🚀 Future Enhancements

If you want to take this project further, consider implementing:
1. **Persistent History**: Write chat history to a JSON file (`history.json`) on exit and reload it on boot to resume previous sessions.
2. **Interactive Selection Menu**: Let the user choose between models (e.g., `llama-3.3-70b-versatile` vs. `mixtral-8x7b-32768`) at startup.
3. **Async IO Streaming**: Re-write the client to use `AsyncGroq` and `asyncio` to allow multiple concurrent streaming sessions (useful if expanding into a web server).
