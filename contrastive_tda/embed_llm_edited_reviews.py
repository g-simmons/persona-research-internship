from embed import embed
from contrastive_tda.catalog import Catalog
from contrastive_tda.schemas import LLMEditedMovieReview, LLMEditedMovieReviewEmbedded
from typing import List

if __name__ == '__main__':
    cat = Catalog()
    llm_edited_reviews = cat.load_llm_edited_movie_reviews() # type: ignore
    embeddings = embed(llm_edited_reviews,"full_prompt","cloud")
    # llm_edited_reviews["full_prompt_embedding"] = embeddings
    embedded_reviews = [LLMEditedMovieReviewEmbedded(**row.model_dump(),full_prompt_embedding=embedding) for (row, embedding) in zip(llm_edited_reviews,embeddings)] # type: ignore
    cat.save_llm_edited_movie_reviews_embedded(embedded_reviews)

# embed the llm edited reviews
# generate the figures
