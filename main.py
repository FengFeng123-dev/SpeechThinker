from src.agent.chat import run_text_mode, run_voice_mode


def main():
    mode = "text"
    while True:
        if mode == "text":
            result = run_text_mode()
            if result == "quit":
                break
            mode = "voice" if result == "voice" else "text"
        elif mode == "voice":
            result = run_voice_mode()
            mode = "text"


if __name__ == "__main__":
    main()
