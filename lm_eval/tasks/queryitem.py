# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""

import os

import pandas as pd
import numpy as np

from lm_eval.base import MultipleChoiceTask
from lm_eval.metrics import queryitem_f1

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class MCTask_Modified(MultipleChoiceTask):
    """A task represents an entire benchmark including its dataset, problems,
        answers, and evaluation methods. See BoolQ for a simple example implementation

        A `doc` can be any python object which represents one instance of evaluation.
        This is usually a dictionary e.g.
            {"question": ..., "answer": ...} or
            {"question": ..., question, answer)
        """
    # TODO: The 'modified' in class name indicates that the data are loaded from a local_dir.

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.

    # DATASET_PATH: str = r"/root/autodl-tmp/projects/llm_relevence_evaluation_data_0523.csv"

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self):
        super().__init__()
        self.DATASET_PATH = r"/root/autodl-tmp/projects/llm_relevence_evaluation_data_0523.csv"

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        assert os.path.exists(self.DATASET_PATH)
        data_df = pd.read_csv(self.DATASET_PATH)
        data_array = data_df.values
        print(data_df.shape)

        np.random.seed(42)
        np.random.shuffle(data_array)
        data_lst = data_array.tolist()[:5000]
        print(len(data_lst))

        self.dataset = data_lst

    class InputTemplate:
        def __init__(self, query: str, title: str, cate_desc: str, label: int):
            self.query = query
            self.title = title
            self.cate_desc = cate_desc
            self.label = label

        def entity(self) -> dict:
            return {
                "id": "None",
                "query": f"[Instruction] Your task is to predict relevance level from 1 to 3 based on the query and the product title and product category:\n"
                         f"1. totally irrevelant product.\n"
                         f"2. the product only partial matches the query.\n"
                         f"3. the product exactly matches query  or contains query consecutively.\n"
                         f"[Query] {self.query}\n"
                         f"[Product]\n"
                         f"Title: {self.title}\n"
                         f"Category: {self.cate_desc}\n"
                         f"[Relevance] ",
                "choices": ["1", "2", "3"],
                "gold": self.label - 1
            }


# TODO: Replace `NewTask` with the name of your Task. 
class QueryItem_Task(MCTask_Modified):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = r"/root/autodl-tmp/projects/llm_relevence_evaluation_data_0523.csv"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__()

    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False

    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return False

    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True

    def training_docs(self) -> list:
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                # self._training_docs = map(self._process_doc, self.data)

                # assert isinstance(data, list)
                data_lst = list(map(self._process_doc, self.dataset))

                return data_lst

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            data_lst = list(map(self._process_doc, self.dataset))

            return data_lst

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            data_lst = list(map(self._process_doc, self.dataset))

            return data_lst

    def _process_doc(self, doc) -> dict:
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        assert isinstance(doc, list) and len(doc) == 4
        query, title, cate_desc, label = doc

        return MCTask_Modified.InputTemplate(query, title, cate_desc, label).entity()

    def doc_to_text(self, doc: dict) -> str:
        # TODO: Format the query prompt portion of the document example.
        return doc["query"]

    """def doc_to_target(self, doc: dict) -> str:
        # TODO: Fill in the `target` ("gold answer") variable.
        # The prepended `" "` is required to space out the `doc_to_text` and
        # `doc_to_target` strings.
        return " " + doc["choices"][doc["gold"]]"""

    # def construct_requests(self, doc, ctx):
    """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
    """

    # TODO: Construct your language model requests with the request factory, `rf`,
    # and return them as an iterable.
    # return []

    def process_results(self, doc, results) -> dict:
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and the corresponding metric result as value
        # for the current `doc`.
        """print(f"results.__class__ == {type(results)}")
        print(results)
        assert False"""
        gold = doc["gold"]
        assert isinstance(gold, int)
        # pick up those correct and relevance of which is 3
        acc = 1.0 if (np.argmax(results) == gold and gold == 2) else 0.0
        precision_denominator = 1 if np.argmax(results) == gold else 0
        recall_denominator = 1 if gold == 2 else 0

        return {
            "qi_f1": (acc, precision_denominator, recall_denominator)
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and an aggregation function as value which
        # determines how to combine results from each document in the dataset.
        # Check `lm_eval.metrics` to find built-in aggregation functions.
        return {"qi_f1": queryitem_f1}

    def higher_is_better(self):
        # TODO: For each (sub)metric in the task evaluation, add a key-value pair
        # with the metric name as key and a `bool` value determining whether or
        # not higher values of that metric are deemed better.
        return {"qi_f1": True}
