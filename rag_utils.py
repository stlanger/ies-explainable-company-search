import torch

import sys
sys.path.append("./loaders")

import math
import time
import gc
import numpy as np

import qdrant_utils
from qdrant_utils import Qdrant

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from config import Config

from tqdm import tqdm

import spacy
from collections import defaultdict

import torch.nn.functional as F


class RAG_Utils:
    
    def __init__(self, config):
        # hyper parameter
        self.qdrant_search_limit = 200
        self.qdrant_min_sim = 0.6
        self.reformulate_sents_n = 10
        self.context_pre_sentence_n = 4
        self.context_post_sentence_n = 5

                
        self.config = config
        self.nlp = spacy.load(config.SPACY_MODEL)                
        
        self._initialize_pipeline()

        # the SentenceTransformer model is fix, since it was used to fill Qdrant with the embeddings of the websites' sentences
        model = SentenceTransformer(config.SENTENCE_TRANSFORMER)
        qdrant_collection_name = config.QDRANT_WEBSITES 
        self.qdrant = Qdrant(config.QDRANT_URL, config.QDRANT_WEBSITES, config.QDRANT_SUMMARIES, model)

    def _initialize_pipeline(self):
        try:
            del self.pipeline            
        except:
            pass

        try:
            del self.tokenizer            
        except:
            pass

        gc.collect() 
        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM)
        
        print("do not _initialize_pipeline for SIGIR eval")       
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model = self.config.LLM,
        #     model_kwargs = {"torch_dtype": torch.bfloat16},
        #     device_map = "cuda",
        #     temperature = 0.001,
        #     top_p = 0.98,
        #     pad_token_id = self.tokenizer.eos_token_id,
        # )
        

    def _reformulate(self, question):
        sys_prompt = "Given a user query describing a problem, need, or desired expertise, generate {self.reformulate_sents_n} example sentences that realistically match "
        sys_prompt += "what companies in the United Kingdom might write on their websites to describe services or offerings relevant to that query. "
        sys_prompt += "Use natural, professional British English in the tone of typical company website copy. "
        sys_prompt += "Focus on how companies describe what they do, what makes their service valuable, and how customers can engage with them. "
        sys_prompt += "The goal is for these sentences to serve as semantically compatible search queries for retrieving matching company website "
        sys_prompt += "content from a vector store. Do not include any introductory or explanatory text—only output the example sentences. "
        sys_prompt += "Do not number the sentences, instead just put each of them on a new line."        
    
        user_prompt = question
    
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=512,
        )
    
        answer = outputs[0]["generated_text"][-1]["content"]    
        doc = self.nlp(answer)
        return [s.text.strip() for s in doc.sents]


    def get_qdrant_utils(self):
        return self.qdrant
    
    
    def _find_similar(self, text):
        candidate_sentences = self._reformulate(text)        
        candidate_sentences.append(text)        
    
        print("\n".join(candidate_sentences), "\n\n")

        embeddings = self.qdrant.embed_texts(candidate_sentences)
        base_query_vector = torch.from_numpy(
            np.array(embeddings)
        ).mean(dim=0)
    
        all_hits = []
        
        for e in embeddings:
            hits = self.qdrant.query_vector(e, limit=self.qdrant_search_limit)
            
            for hit in hits:
                if hit.score <= self.qdrant_min_sim:
                    continue
                company_id = hit.payload["company_id"]
                filepath = hit.payload["filepath"]
                idx = hit.payload["idx"]
                name = hit.payload["name"]
                text = hit.payload["text"]
                web = hit.payload["company_data"][self.config.JSON_COMPANY_URL]
    
                all_hits.append({
                    "company_id": company_id,
                    "name": name,
                    "filepath": filepath,
                    "idx": idx,
                    "text": text,
                    "web": web,
                    "score": hit.score,                
                })

        # we use a dict with the sentence_id as key to get rid of duplicated sentences
        grouped_data = defaultdict(lambda: defaultdict(dict))
        for result in all_hits:
            sentence_id = result['idx']            
            grouped_data[result['company_id']][result['filepath']][sentence_id] = {
                'idx': result['idx'],
                'text': result['text'],
                'name': result['name'],
                'score': result['score'],
                'web': result['web']
            }
    
        grouped_data = {k: dict(v) for k, v in grouped_data.items()}

        # we transform the inner dict (which was used to get rid of duplicates) back to a simple list of entries
        transformed_data = {
            company_id: {
                file_path: list(entries.values())
                for file_path, entries in file_details.items()
            }
            for company_id, file_details in grouped_data.items()
        }

        return (base_query_vector, candidate_sentences, transformed_data)

    
    """
    For a given set of sentence indices, create a list of the form [(context_window_start_idx, context_window_end_idx].
    That is, we substitute a company's most relevant sentences with a context window, created like this:
    - take $self.context_pre_sentence_n$ sentences before the target sentence
    - include the target (similar) sentence
    - take $self.context_post_sentence_n$ after the target sentence
    - merge overlapping windows into one, to get rid of duplicates
    """
    def _create_augmented_company_context_windows(self, company_file_sentence_indices):    
        # sort indices in ascending order
        sorted_indices = sorted(company_file_sentence_indices)
    
        context_window_positions = []
        current_start = None
        current_end = -1
    
        for idx in sorted_indices:
            # calculate current context window
            start = max(0, idx - self.context_pre_sentence_n)
            end = idx + self.context_post_sentence_n + 1
    
            if current_start is None or start > current_end:
                # if current window is the first one or it does not intersect with any other context windows
                if current_start is not None:
                    context_window_positions.append((current_start, current_end))
                current_start = start
                current_end = end
            else:
                # if concext window intersects, change endings
                current_end = max(current_end, end)
    
        # add last context window if it exists
        if current_start is not None:
            context_window_positions.append((current_start, current_end))
    
        return context_window_positions
        

    def _answer(self, query, company_name, context):    
        sys_prompt = '''You are an assistant that evaluates user queries in relation to the context of a company's website. All queries are provided to you in the following format:
"Question: <question> | Name: <name of the company> | Context: <context from the company website>"

The context consists of multiple passages of the company's website, which are separated by the string "############". Based on the context, you assess how relevant the company is for helping me with my question. You assign a relevance score from 0 to 10 stars, where 10 means "very relevant" and 0 means "not relevant". Do not be too optimistic in your rating: 10 stars should only be given if you think that the company is a perfect match to the task described in my query. 8 to 9 stars mean that the company is a very good match. 5 to 7 stars mean the company is relevant but maybe not ideal for my purpose. 3 to 5 stars mean the company might do something similar, but not what I am interested in.

Your response will always begin with the star rating and be followed by a short summary in English, for example:
"3: <summary>"

The summary should be brief and concise, highlighting why you think it might or might not be relevant to my query. Please do not recapitulate the query, just answer to it.
'''

        user_prompt = f'Frage: {query} | Name: {company_name} | Context: {context}'    
    
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )

        # print(">>>>>>>>>>>>>>>>>>> len(user_prompt):", len(user_prompt)) 
    
        answer = outputs[0]["generated_text"][-1]["content"]    
        return answer
        
    
    def query(self, query):
        query = query
        (base_query_vector, candidate_sentences, similar_sents) = self._find_similar(query)
    
        scored_answers = []
        for company_id in tqdm(similar_sents.keys()):
        # for company_id in tqdm(list(similar_sents.keys())[:20]):
            company_data = similar_sents[company_id]
            company_name = ""
            web=""
            
            context = []
            for file in company_data.keys():
                company_name = company_data[file][0]["name"]
                web = company_data[file][0]["web"]
                indices = [entry["idx"] for entry in company_data[file]]                
                
                for (start_idx, end_idx) in self._create_augmented_company_context_windows(indices):
                    context.append(self.qdrant.get_sentence_range(company_id, file, start_idx, end_idx))
        
                context.append("############") 
            contextstring = " ".join(list(context))
            a = self._answer(query, company_name, contextstring[:20000]).strip()        
            scored_answers.append({
                "score": a[0], 
                "company_id": company_id,
                "name": company_name,                
                "website": web, 
                "explanation": a[2:].strip(),
                "context": contextstring,
            })

        summaries = self.qdrant.get_company_summaries([answer["company_id"] for answer in scored_answers])        
        assert(len(summaries) > 0)
        
        summary_vectors = torch.Tensor([s.vector for s in summaries])

        cos_sim = F.cosine_similarity(summary_vectors, base_query_vector, dim=1).tolist()        
        for sim, answer in zip(cos_sim, scored_answers):
            try:
                answer["base_vector_enhanced_score"] = float(answer["score"]) + (sim * 0.1)
            except:
                answer["base_vector_enhanced_score"] = sim * 0.1

        results = sorted(scored_answers, key=lambda x: x["base_vector_enhanced_score"], reverse=True)
        return (candidate_sentences, results)


    
    def rerank_results(self, original_results, upvoted, downvoted, alpha=1, beta=0.6):
        summaries = self.qdrant.get_company_summaries([result["company_id"] for result in original_results])
        assert(len(summaries) > 0)
        
        votes = torch.zeros(1, len(summaries[0].vector))   
        for s in summaries:
            if s.payload["company_id"] in upvoted:
                # print("up")
                votes += alpha * torch.Tensor(s.vector)
            elif s.payload["company_id"] in downvoted:
                # print("down")
                votes -= beta * torch.Tensor(s.vector)
        
        summary_vectors = torch.Tensor([s.vector for s in summaries])
            
        cos_sim = F.cosine_similarity(summary_vectors, votes, dim=1)
        sorted_idx = torch.argsort(cos_sim, descending=True)
    
        results = []
        results = []
        for i in sorted_idx.tolist():        
            payload = summaries[i].payload
            results.append({
                "sim_score": cos_sim[i].item(),
                "company_id": payload["company_id"],
                "company_name": payload["name"],
                "company_summary": payload["summary"]            
            })   
            
        return results

    

    def _recursively_summarize_company_text(self, text, last_word_number=None):
        sys_prompt = "You are tasked with generating a concise summary from the complete text of a company's website, focusing on the most critical information in English."
        sys_prompt += " Your summary should encapsulate an overview of the company's key services and products, highlight unique selling points if available, and briefly mention any notable innovations or recognitions."
        sys_prompt += " Avoid all promotional language and refrain from including website navigation information, disclaimers, or contact details."
        sys_prompt += " Your response should contain only the summary without greetings or introductory phrases and should be entirely in English, regardless of the original language of the input."
        sys_prompt += " Maintain a focus on factual, objective information that accurately represents the company's core operations and distinctive attributes."
        
        user_prompt = text
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=2048,
        )        
        answer = outputs[0]["generated_text"][-1]["content"]  

        doc = self.nlp(answer)
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])

        if word_count<=200 or last_word_number and last_word_number<=word_count:
            return answer
        else:
            print("further shortening...")            
            return self._recursively_summarize_company_text(answer, word_count)

    
    def generate_company_summary(self, company_id, show_company_data=True, max_context_length=200000):
        try:
            company_fulltext = self.qdrant.get_company_fulltext(company_id)[:max_context_length]
            print("generating summary for company:", company_id, "using fulltext length:", len(company_fulltext))
            # print("company_fulltext:", company_fulltext)
            summary = self._recursively_summarize_company_text(company_fulltext)
            if not show_company_data:
                return len(company_fulltext), summary
            else:
                company_data = self.qdrant.get_company_data(company_id)
                company_data["summary"] = summary
                return len(company_fulltext), company_data
        except Exception as e:
            print(e)
            if max_context_length <= 50000:
                raise e
            else:              
                max_context_length *= 0.9
                print(f"context too long for company {company_id} --> cutting context to {max_context_length}")
                self._initialize_pipeline()                
                return self.generate_company_summary(company_id, show_company_data, math.floor(max_context_length))
