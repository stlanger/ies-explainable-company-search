import os
from os import path
import sys

import json
import torch

from datetime import datetime

import pickle
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer

from rag_utils import RAG_Utils

sys.path.append("./loaders")
from qdrant_utils import Qdrant

from config import Config


error_counter = 0

def get_company_dict(json_file_path, json_company_id):    
    company_dict = dict()
    with open(json_file_path) as f:
        data = json.load(f)
        for entry in data:
            company_dict[entry[json_company_id]] = entry
    return company_dict
            

def handle_websitefolder(qdrant, nlp, embedding_model, qdrant_collection_name, json_company_id, json_company_name, websitefolder, companydata):
    no_prev_entries = 0
    
    for file in os.listdir(websitefolder):
        filepath = path.join(websitefolder, file)
        if path.isfile(filepath) and filepath.endswith(".txt"):
            with open(filepath, "r") as f:
                sentences = []
                doc = nlp(f.read()[:1000000])
                sentences = list([sent.text for sent in doc.sents])
                embeddings = embedding_model.encode(sentences)
                
                payloads = [
                    {
                        "filepath": filepath, 
                        "filename": file, 
                        "company_id": companydata[json_company_id], 
                        "name": companydata[json_company_name], 
                        "idx": i, 
                        "text": sentence.strip(), 
                        "company_data": companydata
                    } for i, sentence in enumerate(sentences)
                ]
                
                ids = [(no_prev_entries + i + 1) for i in range(len(payloads))]
                no_prev_entries += len(payloads)

                batch_size = 100
                for i in range(0, len(payloads), batch_size):
                    batch_payloads = payloads[i:i + batch_size]
                    batch_vectors = embeddings[i:i + batch_size]
                    try:
                        qdrant.upsert(qdrant_collection_name, batch_vectors, batch_payloads)                      
                    except Exception as e: 
                        with open(f"logs/error-logs.log", "a") as f:
                            f.write(str(e) + "\n")
                        print(e)
                        error_counter += 1
                        with open(f"logs/error-payloads_{error_counter}.pkl", "wb") as f:
                            pickle.dump(payloads, f)    

def main():
    global error_counter
    cfg = Config("uk")
    rag = RAG_Utils(cfg)
    embedding_model = SentenceTransformer(cfg.SENTENCE_TRANSFORMER)
    qdrant = Qdrant(cfg.QDRANT_URL, cfg.QDRANT_WEBSITES, cfg.QDRANT_SUMMARIES, embedding_model)    
    qdrant.initialize()
    nlp = spacy.load(cfg.SPACY_MODEL)
    company_dict = get_company_dict(cfg.JSON_FILE_PATH, cfg.JSON_COMPANY_ID)
    print(f"Found {len(company_dict)} entries for company_dict in {cfg.JSON_FILE_PATH}")

    error_counter = 0
    
    with open("logs/progress.log", "w") as progress_log:
        print("parsing website folder at:", cfg.WEBSITE_STORAGE_PATH)
        print("writing progress log at: 'logs/progress.log'")
        progress_log.write(str(datetime.now()) + "\n")
        for folder in tqdm(os.listdir(cfg.WEBSITE_STORAGE_PATH), file=progress_log):
        # for folder in tqdm(os.listdir(cfg.WEBSITE_STORAGE_PATH)):
            company_id = "-"
            try:
                websitefolder = path.join(cfg.WEBSITE_STORAGE_PATH, folder)
                if path.isdir(websitefolder):
                    company_id = folder[0: folder.find("_")]
                    company_data = company_dict[company_id]                  
                    handle_websitefolder(
                        qdrant, 
                        nlp, 
                        embedding_model, 
                        cfg.QDRANT_WEBSITES, 
                        cfg.JSON_COMPANY_ID, 
                        cfg.JSON_COMPANY_NAME, 
                        websitefolder, 
                        company_data
                    )
                    fulltext_character_length, summary = rag.generate_company_summary(company_id, False)
                    embeddings = [embedding_model.encode(summary)]
                    payloads = [
                        {                        
                            "company_id": company_id, 
                            "name": company_data[cfg.JSON_COMPANY_NAME],
                            "company_data": company_data,
                            "fulltext_character_length": fulltext_character_length,
                            "summary": summary
                        } 
                    ]
                    qdrant.upsert(cfg.QDRANT_SUMMARIES, embeddings, payloads)   
            except Exception as e:
                print(f"error when processing company {company_id} --> {e}")
                
            

if __name__ == "__main__":
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")
        device = "cuda" if torch.cuda.is_available() else "cpu"    
        print(f"using {device} as device")
    
    main()