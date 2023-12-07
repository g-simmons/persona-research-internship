LLM_EDIT_MOVIE_REVIEWS = contrastive_tda/llm_edit_movie_reviews.py

llm_edited_movie_reviews: $(LLM_EDIT_MOVIE_REVIEWS)
	poetry run python $(LLM_EDIT_MOVIE_REVIEWS)