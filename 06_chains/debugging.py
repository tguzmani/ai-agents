from util.models import build_model, GPT_4_1_MINI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch

model = build_model(GPT_4_1_MINI)

parser = StrOutputParser()


def log_step(x, step_name=""):
    print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
    return x


def debugging(active_method: str):
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | parser

    match active_method:
        # Method 1: Model JSON schema
        case "json_schema":
            print("Chain input schema: ", chain.input_schema.model_json_schema())
            print("Chain input schema: ", chain.output_schema.model_json_schema())

        # Method 2: Use with config
        case "with_config":
            result = chain.with_config(run_name="greeting_chain").invoke(
                {"name": "Alice"}
            )
            print(f"Greeting: {result}")

        case "lambda":
            debug_chain = (
                prompt
                | RunnableLambda(lambda x: log_step(x, "after_prompt"))
                | model
                | RunnableLambda(lambda x: log_step(x, "after_model"))
                | parser
            )

            debug_chain.invoke({"name": "Tu mama cabron"})

        case _:
            valid_methods = ["json_schema", "with_config", "lambda"]
            print("You need to provide a valid debug method, such as:")
            for method in valid_methods:
                print(f"\t{method}")


if __name__ == "__main__":
    debugging("lambda")
