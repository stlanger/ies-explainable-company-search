# SETUP

## 1. Configuration
there are two configurations: 
- `./config/config-de.ini` for German companies
- `./config/config-uk.ini` for companies of the UK

depending on your task, copy the one of the files above to a file "./config/config.ini", which will be used for the project
edit the file and set all necessary paths


## 2. embed website data into Qdrant
This step splits all websites into sentences, embeds them using the sentence transformer model from the config and store them with additional information (file, sentence number, company information) into Qdrant.

1. Attach to the `nlp-gpu-jupyter` docker container (This one runs on the NLP Computer with the 48 GB GPU)
2. inside of the container go to `~/git/ies/retrieval-augmented-exploration/website-rag`
3. run `python3 embed-websites.py`


## 3. running the service



## 4. using the REST-API

The API runs locally, with the ports exposed to the host system at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

It offers the following methods:

#### `submit_task`
- **purpose:** submit a task and add it to the queue
- **request-type:** POST
- **content:** JSON of the form `{"query": <user query as string>}`
e.g.:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "My company was hacked using ransomware. I need someone who can help me to restore my data."}' http://127.0.0.1:5000/submit_task
```

#### `list_tasks`
- **purpose:** list all stored tasks
- **request-type:** GET
- **returns:** a JSON document of the following form:
```JSON
[
    {
        "created": "2025-08-01 06:46:31",
        "id": 2,
        "processing_end": "2025-08-01 06:47:50",
        "processing_start": "2025-08-01 06:46:31",
        "query": "My company was hacked using ransomware. I need someone who can help me to restore my data.",
        "task_id": "c035f34b-20d8-4686-a844-08319a08448c"
    },
    ...

```

e.g.: 
```bash
curl http://127.0.0.1:5000/list_tasks | python3 -m "json.tool"
```

#### `task_results/<task_id>`
- **purpose:** list all results of the task- 
- **request-type:** GET
- **path-parameter `task_id`:** UUID *(make sure to not use the `id` of a task, which is only a number)*
- **returns:** a JSON document of the following form:
```JSON
[
    {
        "explanation": <explanation why a result is (not) relevant for the user>
        "id": <number>,
        "name": <name of the company>,
        "score": <number from 0 (lowest) to 9 (highest)>,
        "task_id": <task_id>,
        "website": <website of the company>
    },
    ...
]
```

e.g.:
```bash
curl http://127.0.0.1:5000/task_details/<task_id> | python3 -m "json.tool"
```

## HELP

#### QDRANT specific
Qdrant is the vector store used for the website sentence embeddings, it runs on a separate docker container `qdrant`. 

- the Qdrant specific setup can be found in `../scripts/qdrant/` and the project's main [README.md](../README.md)
- You can view the dashboard with information about the collections and a simple search at: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
