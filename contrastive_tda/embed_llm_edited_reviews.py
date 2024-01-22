from embed import embed
from contrastive_tda.catalog import Catalog
from contrastive_tda.schemas import LLMEditedMovieReview

if __name__ == '__main__':
    cat = Catalog()
    llm_edited_reviews = cat.load_llm_edited_movie_reviews(as_df=True)
    # print(llm_edited_reviews.dtypes)
    # print(type(llm_edited_reviews.to_dict(orient="records")))
    # print(llm_edited_reviews.to_dict(orient="records")[0])
    embeddings = embed(llm_edited_reviews.apply(lambda x: LLMEditedMovieReview(**x)),"full_prompt","cloud")
    llm_edited_reviews["full_prompt_embedding"] = embeddings
    cat.save_llm_edited_movie_reviews_embedded(llm_edited_reviews)

