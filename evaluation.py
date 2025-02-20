# import os
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from rouge_score import rouge_scorer
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.translate.bleu_score import sentence_bleu
# from retrieval import retrieve_answer_and_reference  # ✅ Import function that fetches both answers

# # Load environment variables
# load_dotenv()

# # Initialize sentence transformer for similarity computation
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Function to compute ROUGE-L score
# def calculate_rouge(reference, candidate):
#     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#     return scorer.score(reference, candidate)['rougeL'].fmeasure

# # Function to compute Cosine Similarity
# def cosine_sim(reference, candidate):
#     ref_emb = embedding_model.encode(reference).reshape(1, -1)
#     cand_emb = embedding_model.encode(candidate).reshape(1, -1)
#     return float(cosine_similarity(ref_emb, cand_emb)[0][0])

# # Function to compute BLEU score
# def calculate_bleu(reference, candidate):
#     return sentence_bleu([reference.split()], candidate.split())

# # ✅ Updated Evaluation Function (Automatically Fetches Reference Answer)
# def evaluate_response_with_rag(query):
#     """
#     Retrieves both reference and chatbot-generated responses and evaluates them.
#     """
#     # ✅ Fetch correct reference answer and chatbot-generated answer from PDFs
#     reference, candidate = retrieve_answer_and_reference(query)

#     # Compute evaluation metrics
#     rouge_score = calculate_rouge(reference, candidate)
#     cosine_score = cosine_sim(reference, candidate)
#     bleu_score = calculate_bleu(reference, candidate)

#     return {
#         "query": query,
#         "reference": reference,
#         "candidate": candidate,
#         "ROUGE-L": rouge_score,
#         "Cosine Similarity": cosine_score,
#         "BLEU": bleu_score
#     }
