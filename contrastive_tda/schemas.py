from pydantic import BaseModel
#pydantic is a library for data validation

class ManualEditedMovieReview(BaseModel):
    original_direction: str # text describing the direction of the film
    original_acting: str # text describing the acting in the film
    original_cinematography: str # text describing the cinematography in the film
    edited_direction: str # sentences describing the direction of a movie, possibly edited from original_direction
    edited_acting: str # sentences describing the acting in a movie, possibly edited from original_acting
    edited_cinematography: str # sentences describing the cinematography in a movie, possibly edited from original_cinematography
    is_direction_edited: str # boolean column indicating whether direction has been edited
    is_acting_edited: str # boolean column indicating whether acting has been edited
    is_cinematography_edited: str # boolean column indicating whether cinematography has been edited

class LLMEditedMovieReview(BaseModel):
    original_direction: str # text describing the direction of the film
    original_acting: str # text describing the acting in the film
    original_cinematography: str # text describing the cinematography in the film
    edited_direction: str # sentences describing the direction of a movie, possibly edited from original_direction
    edited_acting: str # sentences describing the acting in a movie, possibly edited from original_acting
    edited_cinematography: str # sentences describing the cinematography in a movie, possibly edited from original_cinematography
    edited_component: str # component that was edited
    full_prompt: str # concatenation of edited_acting, edited_direction, and edited_cinematography

class LLMEditedMovieReviewEmbedded(BaseModel):
    original_direction: str # text describing the direction of the film
    original_acting: str # text describing the acting in the film
    original_cinematography: str # text describing the cinematography in the film
    edited_direction: str # sentences describing the direction of a movie, possibly edited from original_direction
    edited_acting: str # sentences describing the acting in a movie, possibly edited from original_acting
    edited_cinematography: str # sentences describing the cinematography in a movie, possibly edited from original_cinematography
    edited_component: str # component that was edited
    full_prompt: str # concatenation of edited_acting, edited_direction, and edited_cinematography
    full_prompt_embedding: list[float] # embedding of full_prompt

class SentimentScoredMovieReview(BaseModel):
    original_direction: str # text describing the direction of the film
    original_acting: str # text describing the acting in the film
    original_cinematography: str # text describing the cinematography in the film
    edited_direction: str # sentences describing the direction of a movie, possibly edited from original_direction
    edited_acting: str # sentences describing the acting in a movie, possibly edited from original_acting
    edited_cinematography: str # sentences describing the cinematography in a movie, possibly edited from original_cinematography
    is_direction_edited: str # boolean column indicating whether direction has been edited
    is_acting_edited: str # boolean column indicating whether acting has been edited
    is_cinematography_edited: str # boolean column indicating whether cinematography has been edited
    original_direction_sentiment: str # sentiment score for original_direction
    original_acting_sentiment: str # sentiment score for original_acting
    original_cinematography_sentiment: str # sentiment score for original_cinematography
    edited_direction_sentiment: str # sentiment score for edited_direction
    edited_acting_sentiment: str # sentiment score for edited_acting
    edited_cinematography_sentiment: str # sentiment score for edited_cinematography
    sentiment_scorer: str # name of the model used to calculate sentiment scores