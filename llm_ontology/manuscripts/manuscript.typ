#set page(
  "a4",
  numbering: "1"
)

#set math.equation(
  numbering: "(1)"
)

#show math.equation: set text(size:10pt)

#set text(size: 11pt)

#set par(justify: true)

#align(center)[
#text(size: 22pt)[Ontological Maturity of Large Language Models]
]

#line(length: 100%)
Manuscript

#let todo() = {
  box(height: 0.5em, width: 0.5em, stroke: 1pt + gray)
}

#columns(2)[
= Introduction

- LLMs
  - 
- Ontologies
  - What are they?
_  As reported by Studer et al. [2] from Gruber’s and Borst’s seminal papers [3], [4], “an ontology is a formal, explicit specification of a shared conceptualization”. Conceptualization refers to an abstract model of phenomena in the world produced from the identified relevant concepts of those phenomena. Explicit means that the types of concepts used, and the constraints on their use are explicitly defined. Formal refers to the fact that the ontology should be machine-readable. Shared reflects the understanding that an ontology should capture consensual knowledge accepted by the communities._ - @khadir_ontology_2021
  - How are they used?
- Intersections of LLMs an Ontologies
  - Ontology construction
  - Ontology learning @giglou_llms4ol_2023: "extracting and structuring knowledge from natural language text".
      - Specific tasks:
        - term typing
        - taxonomy discovery
        - extraction of non-taxonomic relations
  - Knowledge Base Question Answering
- Linear Representation Hypothesis
- Park paper @park_geometry_2024
  - shows that ontological structure is reflected in the geometry of the embedding space
- Gap
  - Evaluation in @park_geometry_2024 was only one model
  
#text(fill:gray.lighten(60%))[#lorem(300)]

== Research Questions
// 1. Can we detect ontologies in point clouds, unsupervised?
1. Do better-trained models have better ontology representations?
2. Is better ontology score predictive of better downstream task performance?

== Contributions
- Derive ontological maturity scores based on the linear representation hypothesis
- Score ontological maturity of a variety of models
  - Assess relationship between model size and ontological maturity
  - Assess relationship between model training steps and ontological maturity

  #v(10em)
  
= Background

@park_geometry_2024 and @park_linear_2024 propose the _causal inner product_ as a way to transform the embedding space of a large language model such that it satisfies certain semantic properties. Namely, that dimensions representing causally separable concepts are orthogonal.

Park and coauthors show that the appropriate transformation is given by

$ g(y) arrow.l A(gamma (y) - overline(gamma)_0) $

$ l(x) arrow.l A^(-T) lambda (x) $

for some choice of $A$ and $overline(gamma)_0$.
This choice is non-trivial. Fortunately, Park and coauthors show that
$A$ can be approximated as the whitening of the unembedding layer, and $overline(gamma)_0$ can be approximated as $bb(E) [gamma]$.

Thus $g$ is approximated as

$ g(y) = "Cov"(gamma)^(-1/2)(gamma(y) - bb(E) [gamma]) $

where $gamma$ is the unembedding layer weights, and $bb(E) [gamma] in bb(R)^d$ is the tokenwise mean of these weights, and $d$ is the model embedding dimension.


= Methods
== Ontology Data

*WordNet*

*OBO Foundry* We analyze a set of 261 ontologies from the OBO Foundry. The full list of ontologies is provided in Appendix Table X.

#todo() Check and describe whether ontologies are tree- or DAG-structured.

#todo() Make a table listing the number of terms in each ontology, put it in the Appendix and refer to it here.

== Embedding the Ontology Terms
// We use LLMs to embed the ontologies.
// - Which LLM depends on the experiment
We follow the approach from @park_geometry_2024, describe in Background section of this paper to obtain unembedding vectors for each term in each ontology.

#todo() Are all the ontology terms available in the model tokenizer? Is this required?

== Evaluation
Our aim is to evaluate the ontological maturity of a given model, based on its representations. To do this, we derive three scoring metrics. Each metric captures one desirable property for LLM embedding representations. These properties are:

1. Linear representation
2. Causal separation
2. Hierarchy



#v(33em)

== Multi-Word Expressions
We encountered an issue when attempting to extend the work from Park 2024 for ontologies from other domains. Namely, many of the terms in these other ontologies are multi-word expressions. This is not the case for WordNet, the ontology used in Park 2024. Many of the WordNet terms exist as single tokens in the Gemma model vocabulary, and therefore an unembedding vector for each term can be obtained directly from the model unembedding matrix. This is not the case for multi-word expressions.

We see two approaches to obtain unembeddings for multi-token expressions. The first is to aggregate individual token unembeddings using some function `f: List[unembedding] -> unembedding`, for example the vector average. The second is to learn representations for the unembeddings via gradient descent.

Note that these two approaches could be combined, using aggregation to derive an initial representation for an MWE and using additional training for refinement.

We'll consider the first approach for now, since it's simpler to implement.

=== Approach 1: Aggregating single-token unembeddings to obtain multi-token unembeddings

Given some ontology term $y$ that tokenizes to $n$ tokens $t_y^1, t_y^2, ..., t_y^n$, we obtain an unembedding vector $gamma(y)$ as the vector average of $gamma(t_y^1), gamma(t_y^2), ..., gamma(t_y^n)$.

=== Approach 2: Learning new representations

Adding a token to the model vocabulary requires:
- Updating the tokenizer
- Adding an entry to the embedding table
- Adding an entry to the unembedding table
- Initializing embedding and unembedding representations
  - Note that embedding and unembedding matrices may be tied weights or independent

Initial representations may be refined with additional training.
Sample batches from Pile training data with normalized probabilities.

Sample batches such at P(MWE) is uniform across all MWE's.

=== Linear representation score

  $"linear-rep-score" ("model", "ontology") = bb(E)_"binary-concepts" ("cos"("test-word","concept-vector"))$

  $bb(E)_(y in cal(Y) (w)_"test") "proj"(y,overline(ell)_w)$

  where $overline(ell)_w$ is the LDA vector from Park et. al.:

  $ ell^(‾)_w eq lr((tilde(g)_w^top bb(E) lr((g_w)))) tilde(g)_w $ 

 $ tilde(g)_w eq frac("Cov" lr((g_w))^dagger bb(E) lr((g_w)), norm("Cov" lr((g_w))^dagger bb(E) lr((g_w)))_2) $ 

=== Causal separation score

Causally separable concepts should be represented as orthogonal directions in the embedding space. We operationalize this based on the formalization from @park_geometry_2024.

Given a ground truth ontology the matrix $C$ with entries $c_(W Z) = "cos"(overline(l)_W,overline(l)_Z)$ for pairs of concepts should look similar to the adjacency matrix for the ontology.#footnote([ignoring the diagonal elements, which are 1 for the similarity matrix and 0 for the ontology adjacency matrix.]) However, the language model needs to satisfy multiple goals, so it's unlikely that these will ever be perfectly similar. In the paper it looks like we just want the values where the adjacency matrix is 1 to be higher than the surrounding values.

  // $ "causal-sep-score" (m,o) = \ f("Adj"(o)  -p("cos"(overline(l)_W,overline(l)_Z))) $

  $ "causal-sep-score" (m,o) = \ norm("Adj"(o)  - "off-diag"("cos"(overline(l)_W,overline(l)_Z)))_F $
  
  where $m$ is a model, $o$ is an ontology, $norm(dot)_F$ is the Frobenius norm, and $"off-diag"$ sets the diagonal elements of its argument to 0.

  // #todo() Which matrix norm $f$ to choose? Which processing function $p$ to choose?
  

=== Hierarchy score

The other comparison is on matrix $H$#footnote("H for hierarchy") defined as
$
h_(W Z) = "cos"(overline(l)_W - overline(l)_("par. of " W), overline(l)_Z - overline(l)_("par. of " Z)) $ for all pairs $(W,Z) in O times O$ where $O$ is the set of ontology concepts.


$"cos"(overline(l)_W - overline(l)_("parent of " W), overline(l)_Z - overline(l)_("parent of " Z))$. This one is related to the hierarchy. Want this to be close to identity matrix.

$ "hierarchy-score" = \ 
  norm(bb(1) - "cos"(overline(l)_W - overline(l)_("parent of " W), overline(l)_Z - overline(l)_("parent of " Z)))_F $

where $bb(1)$ is the appropriately-sized identity matrix, 
// $ "hierarchy-score" = \ 
//   f(bb(1) - "cos"(overline(l)_W - overline(l)_("parent of " W), overline(l)_Z - overline(l)_("parent of " Z))) $
  
#todo() Which matrix norm $f$ to choose?

#text(fill:gray.lighten(60%))[#lorem(550)]


== Individual Term Ontology Scores
Given that the matrices used to calculate the scores above consist of representations of individual ontology terms, we propose individual term ontology scores.

[insert some math stuff here idk how]



= Results

== Experiment 2: Ontology score vs. model quality

Our aim in this experiment is to show how ontology score varies with training steps and model size. To evaluate these relationships, we used the Pythia suite of models @biderman_pythia_2023. The Pythia model suite contains model training checkpoints for models with parameter counts {70M, 160M, 410M, 1.0B, 1.4B, 2.8B, 6.9B, 12B} and training steps {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, as well as checkpoints every 1000 steps between 0 and 143,000.

For each combination of model size and training steps ($(142 + 10) times 8 = 1216$ combinations in total), we computed the scores for the WordNet ontology.


=== Experiment 2.1
#text(fill:gray.lighten(0%))[
1. #todo() Does ontology score increase as models get further along in training?

1.2 #todo() Write code that accepts an ontology and a list of Pythia model checkpoint names, downloads the models, embeds each ontology with each model, and stores the embeddings to file.

1.3 #todo() For each embedded ontology, compute its ontology score.

]

#figure(caption: [Ontology scores vs. training steps for different Pythia model sizes],
image("figures/figure-1-sketch.png")
)

series (seaborn param `row`)
- linear-rep-score vs. steps
- causal-sep-score vs. steps
- hierarchy-score vs. steps

marks:
- dot marker to indicate the scores
- lines between dots

channels:
- horizontal position: training steps
- vertical position: ontology score

x-ticks:
- pythia model steps: {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}, as well as checkpoints every 1000 steps between 0 and 143,000.

y-ticks: leave as default

// #figure(caption: [Ontology score vs. model size different Pythia model sizes],
// box(stroke: 1pt + black,height:10em,width:100%))

  
== Experiment 3: Are more ontologically mature models better at downstream tasks?
=== Experiment 3.1
1536xL   Lx2

  - #todo() Select appropriate range of downstream tasks.
    - #todo() Candidates:
      - @giglou_llms4ol_2023
        - term typing
        - taxonomy discovery
        - extraction of non-taxonomic relations
  
  #todo() Check to see what downstream task performances are reported in the Pythia paper.
  
== Experiment 4: Is ontological term frequency correlated with ontological maturity of final model? 
=== Experiment 4.1

Ontology-level:
Ontology-level term frequency: how often do words from the ontology appear in the training corpus?

Ontology-level ontology score: Already defined above

Term-Level:
How often does a specific word from the ontology appear in the training corpus?

Term-level ontology score:
Each term in the ontology appears as a row/column in the symmetric matrices in Figure 5 in Park et. al., and as an x location in Figure 4 from Park et. al.



== Experiment 5: How is unembedding dimensionality correlated with ontology scores?
=== Background
Within the Pythia model suite, models of increasing parameter sizes use larger unembedding matrices. This results in each token being encoded in increasingly higher-dimensional unembedding spaces. For example, the Pythia 70M models encode tokens with 480 dimensions, while the 12B models use 4096 dimensions[fact check this].

This higher dimensionality significantly increases the computation time of ontology scores, making it unreasonable to repeatedly evaluate high parameter models.

For this reason we investigate the effects of reducing dimensionality on the resulting accuracy of ontology scores[rewrite this setnence idk]. The goal of this experiment is to determine if there is a certain dimension for the various Pythia models at which significant computation time is saved while retaining  the original trends of the ontology scores.

=== Methods
Using Principal Component Analysis(PCA) to reduce the dimensionality of the unembedding matrices to various dimensions, ontology scores of 160M and 1.4B parameter models were recalculated every 2000 steps from 1000 to 143000 steps. These scores are then compared with the original scores, taking a mean squared error of the score differences over the respective training steps.

=== Results
The mean squared errors were plotted against the reduced dimension for model sizes 160M and 1.4B.

#figure(caption: [Ontology scores vs. training steps for different Pythia model sizes],
image("figures/baba.png")
)


== Experiment 6: Is term depth correlated with term ontology score?
=== Background
 



= Discussion

Challenges around multiword expressions
#text(gray)[#lorem(70)]

= Conclusion

= Related Work
_ For each paper, take notes on the paper contributions (towards end of intro, discussion, conclusion, abstract). 
 Take notes on research questions.
 Take notes on key distinctions - differences between papers, why it matters._

 ==== The Shape of Word Embeddings @draganov_shape_2024
 - contributions
 -- RQ: do word embedding spaces of a language contain information of its history and structure?
 -- construct algorithms to generate language phylogenetic trees over Indo-European languages. Generating persistence diagrams for each language with persistence homology over 10000 tokens from each langauge, then using the algorithm on those diagrams.
 -- showed that the word embedding space of a language contains info about its history and structure, anad that it can be retrieved through TDA
 
 -key distinctions, compare and contrast
 - Similarities and Differences: Similar to this paper, we are testing if an embedding space contains information of its structure with TDA. Instead of analyzing the embedding space of an entire language with aims of history and structure, we are testing that of a given ontology without its history implications.
   - Why it matters: It could be possible that methods of TDA would be more effective in meeting our goals of extracting structure, as we don't have to worry about history. In theory, ontologies also have a hierarchical structure that we are specifically looking for.

 ==== mOWL: Python library for machine learning with biomedical ontologies
 - RQ: is it possible to map ontology entities into Rn while preserving ontology knowledge, and do models improve on ontology-based tasks when these embeddings are implemented?
 - This work differs as embeddings are used to represent the structural features of an ontology(classes, relations), rather than individual tokens. Methods directly using tokens are mentioned but are concluded to have lower performance than structural embeddings.
]
#pagebreak()
#line(length: 100%)
Notes

= Annotated Bibliography

@gholizadeh_novel_2020

@park_geometry_2024

= References
#bibliography("./zotero.bib", style: "american-psychological-association",title: none)