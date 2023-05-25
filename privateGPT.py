#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import flask

from flask import render_template

app = flask.Flask(__name__)

qa = None

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get("PERSIST_DIRECTORY")

model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

from constants import CHROMA_SETTINGS


@app.route("/app")
def theapp():
    return render_template("app.html", title="App")


@app.route("/")
def index():
    return "Hello, world!"


@app.route("/ask/<string:s>")
def ask(s):
    # return "hello"
    return my_chatGPT_bot(s)


def init():
    print("Okay, starting to bootstrap model!")
    global qa
    if qa is None:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

        print("model_11type = " + model_type)

        match model_type:
            case "LlamaCpp":
                llm = LlamaCpp(
                    model_path=model_path,
                    n_ctx=model_n_ctx,
                    callbacks=[],
                    verbose=False,
                )
            case "GPT4All":
                llm = GPT4All(
                    model=model_path,
                    n_ctx=model_n_ctx,
                    backend="gptj",
                    callbacks=[],
                    verbose=False,
                )
            case _default:
                print(f"Model {model_type} not supported!")
                exit

        print("Prep QA")

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
        )

        print("Okay: bootstrapping model complete!")
    else:
        print("Okay, Model is already bootstrapped and ready to go!")


def my_chatGPT_bot(input_text):
    print("QA the thing")
    init()

    res = qa(input_text)

    print("Response from QA")
    answer = res["result"]

    # Print the result
    print("\n\n> Question:")
    print(input_text)
    print("\n> Answer:")
    print(answer)

    return answer


if __name__ == "__main__":
    # main()
    init()
    app.run()
