from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Any
import torch


class EasyNegativeFinder:
    def __init__(self, embedding_model_name: str, device: str, store_persist_path: str):
        self.embedding_model_name = embedding_model_name
        self.device = device
        self.store_persist_path = store_persist_path

        self.embedding_model = self.__load_embedding_model()
        self.text_splitter = CharacterTextSplitter()
        self.vector_store = None

    def __load_embedding_model(self):
        hf_embedding = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': self.device}
        )
        return hf_embedding

    def __chunk_text(self, texts: List[str]) -> Any:
        chunks = [self.text_splitter.split_text(t) for t in texts]
        return chunks

    def init_vector_store(self, texts: List[str]):
        store = Chroma.from_texts(texts, self.embedding_model,
                                  persist_directory=self.store_persist_path)
        self.vector_store = store

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
