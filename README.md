# germanrag-easyneg

> Finding Easy Negatives in a QA Dataset

The `DiscoResearch/germanrag` datasets comes in the following format: 

```
contexts: [], question: "...", answer: "...", positive_ctx_idx: ...
```

Contexts for each data instance can contain upto 3 hard negatives. Hard Negatives are close to the answer context but for a better QA model, there should be negatives which sound plausible but are not really relevant to the answer context (Easy Negatives).

## Finding Easy Negatives

Since negatives are related to the answer by diminishing similarity, the approach taken in this repo is to find all contexts, create a vector store of them as one would do for a RAG system, then find easy negatives based on their low similarity scores against the supervised answer label.

The vector store is created using langchain and chromadb and for embedding, `sentence-transformers/distiluse-base-multilingual-cased-v2` has been used. One caveat here is that the dataset is for German QA and multilingual sentence-transformers models aren't aplenty. This model serves as an example use case. 

Based on low similarity compared to the answer label, multiple easy negatives are selected, assigned a probability and then sampled as a multinomial distribution. The idea is motivated by negative sampling in Word2Vec that instead of looking at all negatives we sample a smaller portion probabilistically and then use those.


## Preparing data for fine-tuning

Since easy negatives are found, they can be added to the existing contexts and then converted into a suitable format for fine-tuning.

The format followed here is the `context_qa.load_v2` for in context question answering from [GitHub/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#dataset).

### Format

```jsonl
{"context": "...", "question": "...", "answer": "..."}
```

### Caveates

Negative contexts are not supposed to lead to any answers, so for negatives, the answer is set as *Bei dem gegebenen Kontext ist keine Antwort möglich.*

## Example

For a full working example, check the [full_example.ipynb](https://github.com/ShawonAshraf/germanrag-easyneg/blob/main/full_example.ipynb) notebook or for the `main.py` file. 

## Persisting the prepared dataset

Ideally one would save the prepared dataset on disk but this is a small dataset and the repo serves as more of an example. Hence the prepared data is kept in the memory in all the examples. 

## Fine-Tune Config

Check the `ft_config.yml` file.

## One thing about the dataset ...

The Hugginface page for the dataset mentions that the negatives were selected randomly from a set of unique hard negatives. I think the authors missed an opportunity here to sample based on similarity and negative sampling. (Just my two cents!).

## Env Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```
