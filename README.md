# local-model-eval

This repo contains a simple script that measures the performance of [GGUF](https://huggingface.co/docs/hub/en/gguf) LLMs hosted on Huggingface, as executed by the [llama.cpp python bindings](https://github.com/abetlen/llama-cpp-python).  It seems plausible that the performance of the llama-cpp-python is comparable to what you would expect running Ollama, given it is also based on llama.cpp.

Results will be stored in results.json (which will be overwritten after successful execution).

## Limitations
- the [xsum dataset](https://huggingface.co/datasets/EdinburghNLP/xsum) requires local code execution to download the dataset from github
- models being evaluated will be downloaded and stored locally in huggingface_models.  Models can be large and downloading can be slow
- this is intended for use on Macs, but will probably work on other architectures
- this is a quick and dirty attempt to do this, and could be improved in many ways, for example tweaking settings passed to llama.cpp

## Setting options

The list of models that will be evaluated have to be manually updated in `eval.py`.  There are instructions in the script on how to do that.

An example models, `hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF` is already defined by default.

The number of evaluations that are run for each task is configurable as well, see the script for instructions.

## Setup and running

(optional, install [uv](https://github.com/astral-sh/uv))

```
uv run eval.py # add LOGLEVEL=DEBUG for debugging
```

## Evaluations

A few different evaluations are run against the model and statistics are calculated.  For summarization and question answering tasks, you can adjust how many examples from the dataset are executed, obviously the more you run the more accurate the derived statistics, but the slower it will run.

- speed - mean/stdev measurements of tokens per second when performing evaluation tasks
- summarization tasks - instructs the model to summarize texts from the [xsum dataset](https://huggingface.co/datasets/EdinburghNLP/xsum), scored by the [ROUGE metric](https://huggingface.co/metrics/rouge)
- question answering - instructs the model to answer a question from the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/), scored by the [SQuAD metric](https://huggingface.co/spaces/evaluate-metric/squad)

## Example of results.json

```
{
    "system_info": {
        "machine_model": "...example model num",
        "machine_name": "MacBook Pro",
        "model_number": "...example model number",
        "chip_type": "Apple M4",
        "number_processors": "proc 8:4:4",
        "os_loader_version": "...example version",
        "physical_memory": "32 GB"
    },
    "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF": {
        "summary_task_results": {
            "n": 2,
            "rouge_score": {
                "rouge1": 0.27662037037037035,
                "rouge2": 0.09829580036518565,
                "rougeL": 0.1863425925925926,
                "rougeLsum": 0.1863425925925926
            },
            "average_completion_time": 9.200960040092468,
            "average_completion_stddev_time": 7.023865533169396,
            "average_toks_sec": 10.424123496246018,
            "stddev_toks_sec": 3.9613354430693635
        },
        "question_answering_task_results": {
            "n": 2,
            "squad_score": {
                "exact_match": 50.0,
                "f1": 73.52941176470588
            },
            "average_completion_time": 2.7091245651245117,
            "average_completion_stddev_time": 0.9422491066506467,
            "average_toks_sec": 9.27475668685162,
            "stddev_toks_sec": 5.574897672736889
        }
    }
}
```