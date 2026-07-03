from dotenv import load_dotenv

load_dotenv()

from importlib.metadata import version

from langchain_openai import ChatOpenAI

core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    llm = ChatOpenAI(model_name="google/gemini-2.5-flash", temperature=0)
    response = llm.invoke("Answer 'All OK'")
    print(response)


if __name__ == "__main__":
    main()
