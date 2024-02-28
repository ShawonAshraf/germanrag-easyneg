from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Any
import torch
from loguru import logger


class EasyNegativeFinder:
    def __init__(self, embedding_model_name: str, device: str, store_persist_path: str):
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.store_persist_path = store_persist_path

        self.embedding_model = self.__load_embedding_model()
        self.vector_store = None

    def __load_embedding_model(self):
        logger.info(f"Initialising Embeddings with model ::{self.embedding_model_name}")
        hf_embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device}
        )
        logger.success("Done")
        return hf_embedding

    # would've been nicer if langchain had a way
    # to uses a persisted vector store
    # the api has nothing on it, but perhaps can be done manually
    # with chroma api
    # TODO: for future
    def init_vector_store(self, texts: List[str]):
        logger.info("Populating Vector Store")
        logger.info(f"Persisted @ :: {self.store_persist_path}")
        store = Chroma.from_texts(texts, self.embedding_model,
                                  persist_directory=self.store_persist_path)
        self.vector_store = store
        logger.success("Done")

    def find_easy_negs_for(self, data_instance: Any):
        reference = data_instance["answer"]
        # make a set
        # faster lookup using "in"
        contexts = set(data_instance["contexts"])

        # retrieve 20 most relevant docs
        # based on similarity
        # why 20? More samples are better because hard negatives may also come up and we
        # have to filter them
        # hacky solution but makes the job easier for softmax and sampling
        relevant_docs = self.vector_store.similarity_search_with_relevance_scores(reference, k=20)

        # filter out docs which are in contexts
        # we don't want duplicate contexts
        relevant_docs = [doc for doc in relevant_docs if doc[0].page_content not in contexts]

        # prepare for softmax
        scores = torch.tensor([d[1] for d in relevant_docs])
        probas = scores.softmax(dim=-1)
        # sample
        sampled = torch.multinomial(probas, num_samples=1)

        idx = sampled.item()
        return relevant_docs[idx][0].page_content
