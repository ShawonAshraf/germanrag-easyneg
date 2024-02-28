from dataclasses import dataclass
from typing import Any, List
from loguru import logger
from src.docs import EasyNegativeFinder
from tqdm.auto import tqdm


# data class for saving data entries in the context_qa.load_v2 format
# format:  context_qa.load_v2
# {"context": "...", "question": "...", "answer": "..."}
@dataclass
class DataEntry:
    context: str
    question: str
    answer: str


# since each data_instance in the original dataset
# can have multiple hard negatives and also,
# easy negative, adding an "answer not possible from the given context" as answer
# for the negatives (easy and hard)

# also, the data_instance must contain the easy negative in the contexts
# there is no hard check on it, left to the user to add the easy negative
# example: `data_instance["contexts"].append(easy_neg)`
def prepare_jsonl_from_data_instance(data_instance: Any) -> List[DataEntry]:
    contexts = data_instance["contexts"]
    question = data_instance["question"]
    answer = data_instance["answer"]
    pos_ctx_idx = data_instance["positive_ctx_idx"]
    pos_ctx = contexts[pos_ctx_idx]

    new_entries = list()

    for ctx in contexts:
        if ctx == pos_ctx:
            new_entry = DataEntry(ctx, question, answer)
        else:
            # no answer is possible
            msg = "Bei dem gegebenen Kontext ist keine Antwort möglich."
            new_entry = DataEntry(ctx, question, msg)

        # add the dict representation, since jsonl
        # is required
        new_entries.append(new_entry.__dict__)

    return new_entries


# iterate through the entire dataset and create jsonl entries
# inefficient for larger datasets since the list will stay in the memory
# in those cases, an on demand load and convert case (using the function above)
# will be more adequate
def prepare_entire_dataset(dataset: Any, finder: EasyNegativeFinder) -> List[DataEntry]:
    prepared_dataset = list()

    logger.info("Converting Dataset into JSONL format")
    for _, data_instance in tqdm(enumerate(dataset), total=len(dataset)):
        easy_neg = finder.find_easy_negs_for(data_instance)
        data_instance["contexts"].append(easy_neg)

        new_entries = prepare_jsonl_from_data_instance(data_instance)
        prepared_dataset.extend(new_entries)

    logger.success("Done")
    return prepared_dataset
