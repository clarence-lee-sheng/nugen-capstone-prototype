path_vector_db = "/app"

import os
import json
import openai
import sys

import flask

import telebot

from llama_index import VectorStoreIndex, SimpleDirectoryReader, LangchainEmbedding, PromptHelper, LLMPredictor, ServiceContext, set_global_service_context
from llama_index import load_index_from_storage, StorageContext, QuestionAnswerPrompt
from llama_index.llms import OpenAI
from llama_index.indices.postprocessor.node import SimilarityPostprocessor
from IPython.display import Markdown, display

from llama_index import Document
# Create a document with filename in metadata
document = Document(
    text='text',
    metadata={
        'filename': '<doc_file_name>',
        'category': '<category>'
    }
)
document.metadata = {'filename': '<doc_file_name>'}
filename_fn = lambda filename: {'file_name': filename}

from flask import Flask, request, jsonify

from fuzzywuzzy import fuzz, process
from dotenv import load_dotenv

# Set up GPT3.5-Turbo

# set up path which is parent directory of this file
path = os.path.dirname(os.path.abspath(__file__))
load_dotenv(f"{path}/.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens= 1024))
set_global_service_context(service_context)

API_TOKEN = 'NA'

# Prompt will be used to generate rich response,
# Modify this prompt to add instruction

QA_PROMPT_TMPL = (
    "You are DoctorAI and an Ultrasound Physician Assistant. You will use information immediately to suggest an ultrasound imaging procedure appropriate for improving the overall assessment if necessary.\n"
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, Explain and add conclusion at the end: {query_str}\n"
    )
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# Define a list of questions and their corresponding answers

COMMON_QUESTIONS = {
  "Hello.": "Hi, I'm Doctor AI, your private medical advisor. The more details you provide about your symptoms, the more accurate advice I can offer. Please feel free to ask me any questions.",
  "Hi.": "Hi, I'm Doctor AI, your private medical advisor. The more details you provide about your symptoms, the more accurate advice I can offer. Please feel free to ask me any questions."
}

# rebuild storage context

storage_context = StorageContext.from_defaults(path_vector_db=path_vector_db)
# load index. Ensure service_context add here to use customized LLM
index = load_index_from_storage(storage_context=storage_context,service_context=service_context)

# add in prompt template and synthesizer...
query_engine = index.as_query_engine(
    service_context=service_context,
    text_qa_template=QA_PROMPT,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.75)],
    verbose=True,
)

bot = telebot.TeleBot(API_TOKEN)

app = flask.Flask(__name__)

# Process webhook calls

@app.route('/', methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
#        logger.debug(f"Request received: {json_string}")

        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ('', 204)
    else:
        return ('Bad request', 400)

@bot.message_handler(func=lambda message: True)
def echo_message(message):

    # Check if the user typed "/start"
    if message.text == "/start":
        welcome_message = '''
Welcome to EduAI - Your 24/7 Education Advisor!

We are thrilled to have you join us in this revolutionary educational journey. At EduAI, our mission is to provide expert educational support to anyone, anytime, anywhere.

Powered by a state-of-the-art language model enriched with educational knowledge grounded by the relevant course notes, we aim to be your trusted companion for all your health-related inquiries.
        '''
        bot.reply_to(message, welcome_message)
        return

    # Check if the user typed "/start"
    if message.text == "/disclaimer":

        disclaimer_context = '''

DoctorAI Disclaimer

The information provided in response to patient questions on this Telegram channel is generated using a language model that incorporates content from the NHS website, which is publicly available and licensed under the Open Government Licence v3.0.

Please note the following:

Source Attribution: The information provided contains public sector information licensed under the Open Government Licence v3.0. It is derived from the NHS Website Content, but it is not directly cited from the NHS website or endorsed by the NHS specifically.
Content Adaptation: Some of the information presented here may be adapted or modified as part of the language model's responses. While efforts have been made to ensure accuracy, changes in wording or context may occur, which could affect the original meaning or impact of the content.
Risk and Responsibility: Any adaptation of NHS Website Content or use of non-refreshed NHS Website Content may invalidate its formal clinical approval.
Liability: As the provider of this information, I do not bear responsibility for the accuracy, completeness, or validity of the responses generated by the language model.
Seeking Professional Advice: The information provided here should not be considered a substitute for professional medical advice or consultation. If you have specific health concerns or require medical guidance, please consult a qualified healthcare professional or visit official NHS resources.
Independent Verification: It is advisable to independently verify critical information from official NHS sources before making any decisions based on the responses provided on this channel.
Personal Data: No personal data will be collected from the Telegram channel.

By using this Telegram channel, you acknowledge and agree to the above disclaimer. If you do not agree, please refrain from using the information provided here.

        '''

        bot.reply_to(message, disclaimer_context)
        return

    query = message.text
    print(query)

    scores = {}
    for question in COMMON_QUESTIONS:
        score = fuzz.token_sort_ratio(query, question)
        scores[question] = score
    # Find the question with the highest match score
    best_match = process.extractOne(query, COMMON_QUESTIONS.keys())

    # If the match score is above a certain threshold, provide the corresponding answer
    if best_match[1] > 75:
        answer = COMMON_QUESTIONS[best_match[0]]
        bot.reply_to(message, answer)
        return

    bot.reply_to(message, "Message received. Please wait for resposne...")

    response = query_engine.query(query)
    print (response)

    response_ext = Markdown(f"{response.response}").data

    response_ext = response_ext + "\n\nReference used to form this response:\n"

    # Assuming 'response' is the response from your query
    for source_node in response.source_nodes:
        filename = source_node.node.metadata.get('file_name',None)
        filename = filename.split('/')[-1].replace('.txt', '')
        url = "https://www.nhs.uk/conditions/" + filename + "/"
        response_ext = response_ext + url + "\n"

    if response_ext == "None":
        response_ext = "We were unable to find the answer in our current knowledge database. Kindly ask another question."

    bot.reply_to(message, response_ext)

if __name__ == '__main__':
    PORT = int(os.getenv("PORT")) if os.getenv("PORT") else 8080
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)