Usage
=====

Example
-------

The ``BertScore`` class is callable and is the primary way to compute scores.

.. code-block:: python

   from modern_bert_score import BertScore

   candidates = ["Hello World!", "A robin is a bird."]
   references = ["Hi World!", "A robin is not a bird."]

   metric = BertScore(model_id="roberta-base")
   scores = metric(candidates, references)

   # scores is a list of (Precision, Recall, F1) tuples
   for p, r, f1 in scores:
       print(f"P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")

   # To get separate lists of P, R, F1:
   if scores:
       P, R, F1 = zip(*scores)
       print("Precision scores:", P)
       print("Recall scores:", R)
       print("F1 scores:", F1)