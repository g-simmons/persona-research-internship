LLM_EDIT_MOVIE_REVIEWS = contrastive_tda/llm_edit_movie_reviews.py
EMBED_MANUAL_EDITED_MOVIE_REVIEWS = contrastive_tda/manual_edit_movie_reviews.py
EMBED_LLM_EDITED_MOVIE_REVIEWS = contrastive_tda/llm_edit_movie_reviews.py

llm_edited_movie_reviews: $(LLM_EDIT_MOVIE_REVIEWS)
	poetry run python $(LLM_EDIT_MOVIE_REVIEWS)

# embed_llm_edited_reviews:

embed_manual_edited_reviews: $(EMBED_MANUAL_EDITED_MOVIE_REVIEWS)
	poetry run python $(EMBED_MANUAL_EDITED_MOVIE_REVIEWS)

embed_llm_edited_reviews: $(EMBED_LLM_EDITED_MOVIE_REVIEWS)
	poetry run python $(EMBED_LLM_EDITED_MOVIE_REVIEWS)

install_hooks: install_pre_push_hook install_pre_commit_hook

install_pre_push_hook:
	mkdir -p .git/hooks
	cp hooks/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push

install_pre_commit_hook:
    mkdir -p .git/hooks
    cp hooks/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit