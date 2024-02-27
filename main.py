import argparse
from src.docs import EasyNegativeFinder
from loguru import logger
from datasets import load_dataset
from tqdm.auto import trange, tqdm

# create args for model name and persist path
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str,
                    default="sentence-transformers/distiluse-base-multilingual-cased-v2",
                    help="Name of the embedding model from HF hub (ideally a sentence transformer model)")
parser.add_argument("--dpath", type=str, default="./persisted_data/",
                    help="Path to the directory where the vector database will be stored")
parser.add_argument("--device", type=str, default="cpu",
                    help="Choose the device to use for creating embeddings (cpu or cuda)")

args = parser.parse_args()

if __name__ == "__main__":
    DATASET_NAME = "DiscoResearch/germanrag"

    logger.info("Creating Finder Object")
    finder = EasyNegativeFinder(args.model, args.device, args.dpath)

    # load dataset
    logger.info(f"Loading dataset :: {DATASET_NAME}")
    # this dataset has only a train split
    dataset = load_dataset(DATASET_NAME, split="train")

    # find all unique contexts to init the vector store
    logger.info("Loading all unique contexts")
    unique_contexts = set()
    for _, d in tqdm(enumerate(dataset)):
        for ctx in d["contexts"]:
            unique_contexts.add(ctx)
    logger.success(f"Found {len(unique_contexts)} unique contexts")

    # init vector store
    finder.init_vector_store(list(unique_contexts))

    # create easy negatives for five instances
    for idx in trange(5):
        data_instance = dataset[0]
        en = finder.find_easy_negs_for(data_instance)
        print("Current Instance ::")
        print(f"Question : {data_instance['question']}")
        print(f"Answer : {data_instance['answer']}")
        print(f"Easy Negative : {en}")
        print()
