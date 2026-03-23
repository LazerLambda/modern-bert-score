BERTScore
=========

BERTScore, introduced by `Zhang et al., (2019) <https://arxiv.org/abs/1904.09675>`_, leverages pre-trained contextual embeddings from models like BERT to compute a similarity score between candidate and reference sentences.  It addresses limitations of traditional metrics like BLEU and ROUGE by considering semantic vector similarity rather than exact word matches.

**Contextual Embeddings:**

Each token in both the candidate and reference sentences is represented by a contextual embedding vector obtained from a pre-trained language model. These embeddings capture the meaning of a word in its specific context.

**Recall and Precision:**

*   **Recall:**  For each token in the reference sentence, BERTScore finds the most similar token in the candidate sentence based on cosine similarity between their embeddings.  The recall score is the average of these maximum similarity scores for all reference tokens. It reflects how well the candidate sentence covers the information in the reference.
*   **Precision:**  Conversely, for each token in the candidate sentence, BERTScore finds the most similar token in the reference sentence. The precision score is the average of these maximum similarity scores for all candidate tokens. It reflects the proportion of the candidate sentence that is relevant to the reference.

.. image:: ../zhang_19_figure_1.png
   :alt: BERTScore Illustration
   :align: center

*Figure 1 from Zhang et al. (2019) illustrates the BERTScore calculation process.*

**F1 BERTScore:**

The recall and precision scores are combined into a single F1 score, which is the harmonic mean of recall and precision. This F1 score, referred to as BERTScore, provides a balanced measure of the similarity between the candidate and reference sentences.

.. math::
    BERTScore = F_1 = \frac{2 \cdot P \cdot R}{P + R}

**IDF Weighting (Optional):**

BERTScore can optionally incorporate IDF (Inverse Document Frequency) weighting to give more importance to rare words.  Tokens are weighted based on their frequency in a large corpus. This helps to focus on more informative words when calculating similarity.

**Baseline Rescaling:**

Since BERTScore lies in the range of cosine similarity scores between [-1, 1], baseline rescaling can be applied to map the score to [0, 1]. This involves subtracting a baseline score (computed on a large set of sentence pairs) from the original BERTScore.
