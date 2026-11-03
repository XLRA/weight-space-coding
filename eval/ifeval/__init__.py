"""The official IFEval instruction checkers (Google Research, Apache 2.0), as vendored by
EleutherAI/lm-evaluation-harness (lm_eval/tasks/ifeval, commit f724be699e8a, MIT; see LICENSE-lm-eval.md).
Changed for /opt/wsp: the imports point at this package, and the sentence tokenizer reads NLTK's
punkt_tab instead of the pickled punkt model. Nothing else is modified.
"""
