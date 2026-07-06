from smart_qa_bot import SmartQABot

bot = SmartQABot()


def print_header():
    print("=" * 60)
    print("SMART Q&A BOT DEMO")
    print("=" * 60)


def bot_ask_one_loop():

    questions = [
        "What is Linux?",
        "Why is that Mac OS cannot run on PC/Linux hardware?",
        "Why is LangSmith better than LangFuse?",
    ]

    print_header()

    for question in questions:
        response = bot.ask(question)

        print(f"Question: {question}")
        print("-" * 60)

        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence.value}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {response.follow_up_questions}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)


def bot_error_handling():
    long_question = "What is very" + "hurr durr " * 100 + "important?"

    response = bot.ask(long_question)

    print(f"Handled gracefully: {response}")


def bot_ask_batch_loop():
    questions = [
        "What is Python?",
        "What is Javacript?",
        "What is Rust?",
    ]

    responses = bot.ask_batch(questions)

    for question, response in zip(questions, responses):
        print(f"Question: {question}")
        print(f"Response: {response}")
        print()


if __name__ == "__main__":
    try:
        # bot_ask_one_loop()
        # bot_ask_batch_loop()
        # bot_error_handling()
        pass
    finally:
        bot.flush()
