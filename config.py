import os
from configparser import ConfigParser

class Config:    
    def __init__(self, country_code):
        
        # language code needs to either be "uk" or "de"
        if country_code not in ["de", "uk"]:            
            raise Exception('You need to use "de" or "uk" for the configurator')
        
        cfg = ConfigParser()
        module_dir = os.path.dirname(__file__)
        cfg.read(f"{module_dir}/config/config-{country_code}.ini")
        
        print("using the following configuration:\n")
        
        self.WEBSITE_STORAGE_PATH = cfg["STORAGE"]["PATH"]
        
        self.QDRANT_URL = cfg["QDRANT"]["URL"]
        self.QDRANT_WEBSITES = cfg["QDRANT"]["COMPANY_WEBSITES"]
        self.QDRANT_SUMMARIES = cfg["QDRANT"]["COMPANY_SUMMARIES"]
        
        self.JSON_FILE_PATH = cfg["JSON_DATA"]["PATH"]
        self.JSON_COMPANY_ID = cfg["JSON_DATA"]["COMPANY_ID"]
        self.JSON_COMPANY_NAME = cfg["JSON_DATA"]["COMPANY_NAME"]
        self.JSON_COMPANY_URL = cfg["JSON_DATA"]["COMPANY_URL"]

        self.SQLITE_FILEPATH = cfg["SQLITE"]["PATH"]
         
        self.SPACY_MODEL = cfg["NLP"]["SPACY_MODEL"]
        self.SENTENCE_TRANSFORMER = cfg["NLP"]["SENTENCE_TRANSFORMER"]
        self.LLM = cfg["NLP"]["LLM"]
        
        

        print("The Config objects offers the following variables")
        print("------ WEBSITE STORAGE ----------")
        print("WEBSITE_STORAGE_PATH   ", self.WEBSITE_STORAGE_PATH)
        print()
        print("------ QDRANT VECTOR STORE ------")
        print("QDRANT_URL url:        ", self.QDRANT_URL)
        print("QDRANT_WEBSITES:     ", self.QDRANT_WEBSITES)
        print("QDRANT_SUMMARIES:     ", self.QDRANT_SUMMARIES)
        print()
        print("------ JSON_DATA ----------------")
        print("JSON_FILE_PATH:        ", self.JSON_FILE_PATH)
        print("JSON_COMPANY_ID:       ", self.JSON_COMPANY_ID)
        print("JSON_COMPANY_NAME:     ", self.JSON_COMPANY_NAME)
        print("JSON_COMPANY_URL:     ", self.JSON_COMPANY_URL)
        print()
        print("------ SQLITE -------------------")
        print("SQLITE_FILEPATH:       ", self.SQLITE_FILEPATH)
        print()
        print("------ NLP ----------------------")
        print("SPACY_MODEL:           ", self.SPACY_MODEL)
        print("SENTENCE_TRANSFORMER:  ", self.SENTENCE_TRANSFORMER)
        print("LLM:                   ", self.LLM)
        print()