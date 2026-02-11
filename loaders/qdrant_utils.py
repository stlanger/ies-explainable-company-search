import sys

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.models import Distance, VectorParams, FieldCondition, MatchValue, MatchAny, Filter, Range
import uuid

class Qdrant:
    
    def __init__(self, url, websites_collection, summaries_collection, embedding_model):
        self.qdrant = QdrantClient(url, timeout=600000)
        
        self.websites_collection = websites_collection
        self.summaries_collection = summaries_collection
        self.embedding_model = embedding_model


    #####################################################################
    #### initialize vector store: (re-)create collection
    
    def initialize(self):
        self.qdrant.recreate_collection(
            collection_name = self.websites_collection,
            vectors_config=VectorParams(
                size = self.embedding_model.get_sentence_embedding_dimension(), 
                distance = Distance.COSINE,
                on_disk = True,
            ),
        )
        self._create_index(self.websites_collection, "company_id", "keyword")
        self._create_index(self.websites_collection, "company_name", "keyword")
        self._create_index(self.websites_collection, "idx", "integer")
        self._create_index(self.websites_collection, "filepath", "keyword")

        self.qdrant.recreate_collection(
            collection_name = self.summaries_collection,
            vectors_config=VectorParams(
                size = self.embedding_model.get_sentence_embedding_dimension(), 
                distance = Distance.COSINE,
                on_disk = True,
            ),
        )
        self._create_index(self.summaries_collection, "company_id", "keyword")
        self._create_index(self.summaries_collection, "company_name", "keyword")
        self._create_index(self.websites_collection, "fulltext_character_length", "integer")

    
    def _create_index(self, collection_name, field, type):
        self.qdrant.create_payload_index(
            collection_name = collection_name,
            field_name = field,
            field_schema = type
        )

    #####################################################################
    #### initialize vector store for company sumarries: (re-)create collection

    def initialize_summaries(self):
        self.qd

    #####################################################################
    #### insertion methods

    def upsert(self, collection_name, batch_vectors, batch_payloads):
        self.qdrant.upsert(
            collection_name = collection_name,
            points = Batch(
                ids = [str(uuid.uuid4()) for i in range(len(batch_vectors))],
                vectors = batch_vectors,
                payloads = batch_payloads
            )
        )
    

    
    #####################################################################
    #### retrieval methods

    def embed_texts(self, texts):
        return [self.embedding_model.encode(text) for text in texts]
    
    def query_text(self, text, limit=10000):
        emb = self.embedding_model.encode(text)
        hits = self.qdrant.query_points(
            collection_name=self.websites_collection,
            query = emb,
            limit = limit
        ).points
        return hits

    def query_vector(self, vector, limit=0000):        
        hits = self.qdrant.query_points(
            collection_name=self.websites_collection,
            query = vector,
            limit = limit
        ).points
        return hits

    #####################################################################
    #### filtering methods
    
    def get_sentence(self, company_id, filepath, idx):
        hits = self.qdrant.query_points(
            collection_name=self.websites_collection,
            limit=1,
            query_vector=[0.0] * self.embedding_model.get_sentence_embedding_dimension(), # define an alibi vector, which is not really used
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=company_id)
                    ),
                    FieldCondition(
                        key="filepath",
                        match=MatchValue(value=filepath)
                    ),                
                    FieldCondition(
                        key="idx",
                        match=MatchValue(value=idx)
                    )
                ]
            ),
            search_params = {
                
            }
        )
        if len(hits) == 0:
            return None
    
        return hits[0].payload["text"]

   
    def get_sentence_range(self, company_id, filepath, idx_start, idx_end, text_join_string=" "):        
        hits = self.qdrant.scroll(
            collection_name=self.websites_collection,
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=company_id)
                    ),
                    FieldCondition(
                        key="filepath",
                        match=MatchValue(value=filepath)
                    ),                
                    FieldCondition(
                        key="idx",
                        range=Range(
                            gt=None,
                            gte=idx_start,
                            lt=idx_end,
                            # lt=None,
                            lte=None,
                        ),
                    )
                ]
            ),
            limit=sys.maxsize,
            with_payload=True,
            with_vectors=False
        )
        sorted_data = sorted([hit for hit in hits[0] if "idx" in hit.payload], key=lambda hit: hit.payload["idx"])
        sents = [data.payload["text"] for data in sorted_data]
        return text_join_string.join(sents)

    
    """
    get all text from the stored website of a company
    """
    def get_company_fulltext(self, company_id, text_join_string=" "):
        # get all sentences, limit by max integer
        results = self.qdrant.scroll(
            collection_name=self.websites_collection,            
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=company_id)
                    )
                ]
            ),         
            limit=sys.maxsize,
            with_payload=True,
            with_vectors=False,
        )        

        # in order to get meaningful text: order by filepath, idx
        sorted_data = sorted(results[0], key=lambda record: (record.payload['filepath'], record.payload['idx']))
        sents = text_join_string.join([record.payload["text"] for record in sorted_data])        
        return sents


    def get_company_data(self, company_id):
        results = self.qdrant.scroll(
            collection_name=self.websites_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=company_id)
                    )
                ]
            ),         
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return results[0][0].payload["company_data"]


    def get_company_summaries(self, company_ids):
        # print(f"finding summaries for {len(company_ids)} companies.")
        results = self.qdrant.scroll(
            collection_name=self.summaries_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="company_id",
                        match=MatchAny(any=company_ids)
                    )
                ]
            ),         
            limit=sys.maxsize,
            with_payload=True,
            with_vectors=True,
        )
        return results[0]
        
    
    def augment_context(self, company_id, filepath, idx, pre_sent_n=3, post_sent_n=5, text_join_string=" "):
        text = self.get_sentence_range(company_id, filepath, max(0, idx-pre_sent_n), idx+post_sent_n+1, text_join_string)
        return text

