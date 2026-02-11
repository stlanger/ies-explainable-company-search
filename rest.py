import os
import sys

import json
import re
import torch
import numpy
import uuid
import random


from flask import Flask, request, jsonify, g
from flask_cors import CORS
from threading import Thread
from queue import Queue
import time
import uuid
import sqlite3
from rag_task_storage import RAG_Task_Storage

from config import Config
from rag_utils import RAG_Utils

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

device = "cuda" if torch.cuda.is_available() else "cpu"



#############################################################################################
### BASIC INITIALIZATION ####################################################################
#############################################################################################

sys_prompt = 'You are an assistant that evaluates user queries in relation to the context of a company\'s website. '
sys_prompt += 'All queries are provided to you in the following format:\n'
sys_prompt += '"Question: <question> | Name: <name of the company> | Context: <context from the company website>"\n\n'
sys_prompt += 'Based on the context, you assess how relevant the company is for the user\'s query. '
sys_prompt += 'You assign a relevance score from 0 to 9 stars, where 9 means "highly relevant" and 0 means '
sys_prompt += '"not relevant at all". Your response should always begin with the star rating and be followed by '
sys_prompt += 'a short summary, for example: \n"3: <summary>"\n\n'
sys_prompt += 'The summary should be brief and concise, highlighting the key facts relevant to the user\'s query.'

print(sys_prompt)



cfg = Config("uk")
rag = RAG_Utils(cfg)




def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(cfg.SQLITE_FILEPATH)
    return g.db


# Initialize Flask app
app = Flask(__name__)
CORS(
    app,
    origins=["http://localhost:5173", "http://localhost:5174"],
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True,
)

# Task storage and queue
task_store = RAG_Task_Storage(get_db)

task_queue = Queue()  # Queue for pending tasks


with app.app_context():
    """Initialize the database when the app starts."""
    task_store.initialize_database()
    

@app.teardown_appcontext
def close_db(exception):
    """Close the database connection at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


# Background worker to process tasks sequentially
def task_worker():
    worker_db_path = cfg.SQLITE_FILEPATH
    worker_task_store = RAG_Task_Storage(lambda: sqlite3.connect(worker_db_path))
    worker_task_store.initialize_database()

    for task in worker_task_store.list_unfinished_tasks():
        task_queue.put((task.get("task_id"), task.get("query")))

    while True:
        task_id, query = task_queue.get()

        worker_task_store.set_start_time(task_id)
        (candidate_sentences, results) = rag.query(query)  
        worker_task_store.insert_task_candidate_sentences(task_id, candidate_sentences)
        worker_task_store.insert_results(task_id, results)
        worker_task_store.set_end_time(task_id)

        task_queue.task_done()


# Start the background worker thread
worker_thread = Thread(target=task_worker, daemon=True)
worker_thread.start()


#############################################################################################
### Standard API Calls ######################################################################
#############################################################################################

# Endpoint to submit a new task
@app.route('/submit_task', methods=['POST'])
def submit_task():
    print("submit_task deactivated for SIGIR evaluation")
    # data = request.get_json()
    # print("data:", data);
    # if not data or not isinstance(data.get('query'), str):
    #     return jsonify({'error': 'Invalid input: query must be a string'}), 400;

    # query = data['query']
        
    # task_id = str(uuid.uuid4())
    # task_store.insert_task(task_id, query)
    
    # task_queue.put((task_id, query))

    # return jsonify({'task_id': task_id, 'status': 'Queued', 'query': query}), 202
    return jsonify({}), 202
    



# Endpoint to list all tasks
@app.route('/list_tasks', methods=['GET'])
def list_tasks():
    return jsonify(task_store.list_tasks())

@app.route('/task_details/<task_id>', methods=['GET'])
def task_details(task_id):    
    return jsonify(task_store.task_details(task_id))
    


# Endpoint to check task status
@app.route('/task_results/<task_id>', methods=['GET'])
def task_status(task_id):
    min_score = request.args.get('min_score', type=int, default=0)
    print("min_score:", min_score)
    
    results = task_store.list_results_for_task(task_id)
    summaries = rag.qdrant.get_company_summaries([result["company_id"] for result in results])
    
    filtered_results = [result for result in results if result['score'] >= min_score]
    summaries = rag.qdrant.get_company_summaries([result["company_id"] for result in filtered_results])
    summaries_map = { s.payload["company_id"]: s.payload["summary"] for s in summaries }
    
    for result in filtered_results:
        result["company_summary"] = summaries_map[result["company_id"]]
    
    if len(filtered_results) == 0:
        return jsonify({'error': f'no results for task with id: {task_id}'}), 404
    
    return jsonify(filtered_results)



# Endpoint to re-order an esisting task
@app.route('/rerank_task_results', methods=['POST'])
def rerank_task_results():    
    data = request.get_json()    
    results = task_store.list_results_for_task(data["task_id"])
    reranked = rag.rerank_results(results, data["upvotes"], data["downvotes"])
    
    return jsonify(reranked)




#############################################################################################
### Evauation API Calls #####################################################################
#############################################################################################


@app.route('/create_user_session/<type_id>', methods=['GET'])
def create_user_session(type_id):
    session_id = str(uuid.uuid4())
    fixed_task_ids = ['99803dd3-8238-4982-9a4c-87dfdcf3f303', '6a502185-a6d9-465a-a2f4-5614b7df4e99', '6c63b400-f9bc-49e1-a7a3-a6a8cd689f80']
    fixed_tasks = [task_store.task_details(id) for id in fixed_task_ids]    
    tasks = task_store.list_tasks(blacklist=fixed_task_ids)
    random.shuffle(tasks)

    task_list = fixed_tasks + tasks[:3]
    task_store.insert_session(session_id, type_id)
    task_store.insert_session_tasks(session_id, [task["task_id"] for task in task_list])
    
    return { 
        "session_id": session_id,
        "tasks": task_list
    }


@app.route('/user_feedback/', methods=['POST'])
def user_feedback():
    data = request.get_json()
    print("user_feedback:", data)
    task_store.insert_user_feedback(data["session_id"], data["task_id"], data["feedback"])
    return jsonify(True)


#############################################################################################
### Start API ###############################################################################
#############################################################################################

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)