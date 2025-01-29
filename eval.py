import os, json, time, logging, subprocess
from typing import Optional

from llama_cpp import Llama
from datasets import load_dataset
from evaluate import load as load_metric
from statistics import mean, stdev

# This script pulls down a set of hugging face GGUF models, evaluates their performance (speed, performance on tasks), writes results to a json file
#
# to add a new model, go to hugging face and find a gguf model, select use this model on the model card, select llama-cpp-python, get repoid and filename
# ...then paste below, the model will be downloaded and cached if its not already in ~/huggingface_models

MODEL_LIST = {
    "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF": "llama-3.2-3b-instruct-q8_0.gguf",
    # "unsloth/phi-4-GGUF": "phi-4-F16.gguf",
    # "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF": "DeepSeek-R1-Distill-Qwen-1.5B-Q2_K.gguf",
    # "TheBloke/deepseek-coder-6.7B-instruct-GGUF": "deepseek-coder-6.7b-instruct.Q2_K.gguf",
}

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(message)s")

# number of evaluations to run, must be >= 2 for means and stddevs
NUM_SUMMARIES = 2
NUM_QA = 2

# get datasets
summary_dataset = load_dataset("xsum", split="test[:1%]", trust_remote_code=True) 
squad_dataset = load_dataset("squad", split="train")

# directory to cache Hugging Face models
cache_dir = "./huggingface_models"

# get metrics
rouge_metric = load_metric("rouge")
squad_metric = load_metric("squad")

def timed_completion(llm: Llama.model, content: str, response_format: Optional[dict] = None):
  "perform completion on llm with given content, returning prediction and statistics"
  response_format = response_format or {"type": "text"}
  start_time = time.time()
  output = llm.create_chat_completion(
    messages = [
      {
        "role": "user",
        "content": content
      }
    ],
    response_format=response_format,
  )
  end_time = time.time()
  completion_time = end_time - start_time
  prediction = output['choices'][0]['message']['content']
  if response_format["type"] == "json_object":
    prediction = json.loads(prediction)
  pred_tokens = output['usage']['completion_tokens']
  result = {
    "prediction": prediction,
    "pred_tokens": pred_tokens,
    "time": completion_time,
    "toks_sec": pred_tokens / completion_time
  }
  return result
  
def evaluate_summarization(llm: Llama.model, dataset, num_items: int = 2):
  "perform N summarizations, computing statistics and returning the results"
  logging.debug(' - evaluating summarization')
  # load summary dataset
  sources = dataset["document"][0:num_items]
  ground_truth_summaries = dataset["summary"][0:num_items] # ground truth summaries

  # generate summaries using the llm
  results = []
  for text in sources:
    result = timed_completion(llm, f"Summarize the following text in a few sentences.  Only return the summary, no additional text:\n\n{text}")
    results.append(result)

  # calc average and stddev
  times = [entry["time"] for entry in results]
  average_time = mean(times)
  stddev_time = stdev(times)
  
  toks_sec = [entry["toks_sec"] for entry in results]
  average_toks_sec = mean(toks_sec)
  stddev_toks_sec = stdev(toks_sec)

  times = [entry["time"] for entry in results]
  
  # compute rouge
  predictions = [entry["prediction"] for entry in results]
  rouge_score = rouge_metric.compute(predictions=predictions, references=[[r] for r in ground_truth_summaries])
  stats = {
    "n": num_items,
    "rouge_score": rouge_score,
    "average_completion_time": average_time,
    "average_completion_stddev_time": stddev_time,
    "average_toks_sec": average_toks_sec,
    "stddev_toks_sec": stddev_toks_sec
  }
  return stats

def evaluate_question_answering(llm: Llama.model, dataset, num_items: int = 5):
  "perform N question answering tasks, computing statistics and returning the results"
  logging.debug(' - evaluating question answering')
  squad_dataset = dataset.take(num_items)
  
  def add_id_to_answer(i: dict):
    "inline function to convert squad dataset answers into a format squad metric can handle"
    id = i['id']
    answer = { 'answers': i['answers'], 'id': id }
    return answer

  results = []
  for d in squad_dataset:
    context, question, answer, id = d['context'], d['question'], d['answers']['text'], d['id']
    result = timed_completion(
      llm,
      f"You are a question answering assistant, answer only using the context given.\nContext: {context}\nQuestion: {question}",
      {
        "type": "json_object",
        "schema": {
          "type": "object",
          "properties": {"answer": {"type": "string"}},
          "required": ["answer"],
        },
      }
    )
    # special handling for later squad comparison
    result['prediction_text'] = {'prediction_text': result["prediction"]["answer"], 'id': id}
    logging.debug(f" - question {question}, gold answer: {answer}, prediction {result["prediction"]["answer"]}")
    results.append(result)

  # calc average and stddev
  times = [entry["time"] for entry in results]
  average_time = mean(times)
  stddev_time = stdev(times)
  
  toks_sec = [entry["toks_sec"] for entry in results]
  average_toks_sec = mean(toks_sec)
  stddev_toks_sec = stdev(toks_sec)

  times = [entry["time"] for entry in results]

  predictions = list(map(lambda r: r['prediction_text'], results))
  answers = list(map(add_id_to_answer, squad_dataset))
  qa_score = squad_metric.compute(predictions=predictions, references=answers)
  stats = {
    "n": num_items,
    "squad_score": qa_score,
    "average_completion_time": average_time,
    "average_completion_stddev_time": stddev_time,
    "average_toks_sec": average_toks_sec,
    "stddev_toks_sec": stddev_toks_sec
  }

  return stats

def evaluate_model(repo_id: str, file_name: str, summary_dataset: dict[str, list[str]]):
    logging.info(f" - Loading model")
    llm = Llama.from_pretrained(
      repo_id=repo_id,
      filename=file_name,
      n_ctx=4048, # max context size default is 512?!
      verbose=logging.root.level < logging.INFO
    )
    # run a single completion, to ensure the model has been downloaded
    llm.create_chat_completion(
      messages = [
        {
          "role": "user",
          "content": "return value of 1+1"
        }
      ],
    )
    logging.info(f' - Successfully downloaded, running evaluations')

    summary_result = evaluate_summarization(llm, summary_dataset, NUM_SUMMARIES)
    qa_result = evaluate_question_answering(llm, squad_dataset, NUM_QA)
    return { "summary_task_results": summary_result, "question_answering_task_results": qa_result }

def get_mac_details():
  "return select mac stats"
  try:
    output = subprocess.check_output(['system_profiler', 'SPHardwareDataType', '-json']).decode()
    sysinfo = json.loads(output)['SPHardwareDataType'][0]
    return {
        'machine_model': sysinfo['machine_model'],
        'machine_name': sysinfo['machine_name'],
        'model_number': sysinfo['model_number'],
        'chip_type': sysinfo['chip_type'],
        'number_processors': sysinfo['number_processors'],
        'os_loader_version': sysinfo['os_loader_version'],
        'physical_memory': sysinfo['physical_memory']
    }
  except Exception as e:
    print(f"Error getting Mac make and model: {e}")
    return 'unknown'
    
if __name__ == "__main__":
    results = {}
    for repo_id, file_name in MODEL_LIST.items():
        logging.info(f"Evaluating {repo_id}...")
        evaluation = evaluate_model(repo_id, file_name, summary_dataset)
        results['system_info'] = get_mac_details()
        results[repo_id] = evaluation

    # save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info('Evaluation complete. Results saved to results.json')
