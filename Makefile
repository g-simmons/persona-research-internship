LLM_EDIT_MOVIE_REVIEWS = contrastive_tda/llm_edit_movie_reviews.py
EMBED_LLM_EDITED_MOVIE_REVIEWS = contrastive_tda/llm_edit_movie_reviews.py

llm_edited_movie_reviews: $(LLM_EDIT_MOVIE_REVIEWS)
	poetry run python $(LLM_EDIT_MOVIE_REVIEWS)

# embed_llm_edited_reviews:

embed_manual_edited_reviews: $()
	poetry run python $()

install_hooks: 
	mkdir -p .git/hooks
	cp hooks/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push