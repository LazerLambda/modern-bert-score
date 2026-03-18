Installation
============

Modern-BERT-Score comes in **two variants**: a base version and a vLLM-enhanced version. For vLLM, an NVIDIA GPU is strongly recommended.

Base Version
------------

.. code-block:: bash

   pip install modern-bert-score

vLLM Version
------------

.. code-block:: bash

   pip install modern-bert-score[vllm]

This implementation is significantly faster than the original BERTScore, especially with GPU acceleration.