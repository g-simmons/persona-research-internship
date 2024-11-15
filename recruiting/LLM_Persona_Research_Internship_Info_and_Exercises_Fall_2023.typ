#set page(numbering: "1")
#set text(
  font: ("Helvetica Neue",),
  size: 12pt,
  // fill: blue,
)

#let ul = underline
#let small(body) = { text(size: 0.8em)[#body] }
#show link: underline
#set par(justify: true)
#set line(length: 100%, stroke: 0.5pt)
#show heading.where(level: 2): it => block(width: 100%)[
  // #set align(center)
  #set text(1.5em)
  #it.body
  #v(0.5em)
]

#show heading.where(level: 1): it => block(width: 100%)[
  // #set align(center)
  #set text(1.6em)
  #it.body
  #v(0.5em)
]

#show heading.where(level: 3): it => block(width: 100%)[
  // #set align(center)
  #set text(1.4em)
  #it.body
  #v(0.5em)
]

#show heading.where(level: 4): it => block(width: 100%)[
  // #set align(center)
  #set text(1.25em)
  #it.body
  #v(0.5em)
]
#show heading.where(level: 5): it => block(width: 100%)[
  // #set align(center)
  #set text(1.125em)
  #it.body
  #v(0.5em)
]

// = LLM Persona Research Internship

// #v(-1em)
// #text(size: 1.6em, weight: "bold")[Fall 2023]
// #v(-1em)
// Gabriel Simmons
// #v(-1em)
// #link("gsimmons@ucdavis.edu")
// #v(1.5em)

// #line(length: 100%, stroke: 0.5pt)

// == About This Internship

// === Topics

// My research is currently focused on Large Language Model (LLM) personas.

// // *The technical perspective*

// #table(
//   columns: (auto, auto),
//   stroke: none,
//   [*The Technical Perspective*],
//   [*The Social Perspective*],
//   [ - To what extent can Large Language Models simulate human behavior, cognition, and opinion formation?
//   - What properties of LLMs are important for this ability?
//   - Can we find parsimonious explanations for how LLMs achieve this behavior?
//   - How can we reliably evaluate this ability?
//   - What are the appropriate formalisms to describe this ability?
//   ],
//   [- What happens in society when LLMs and other AI models become more proficient at simulating humans?
//   - What should we be doing to prepare?
//   ],
// )

// === General Info

// #table(
//   columns: (50%, auto),
//   stroke: none,
//   [*This internship is:*],
//   [*I am:*],
//   [- Unpaid
//   - Informal
//   - Remote-first hybrid],
//   [
//   - junior in my own research career
//   - not a PhD
//   - opinionated about the research process],
// )

// #block[
//   The fact that I'm very early in my own research career is important to consider. Here are some possible advantages and disadvantages of working with me:

//   #table(
//     columns: (auto, auto),
//     // inset: 0.5em,
//     column-gutter: 1em,
//     stroke: none,
//     [*Possible Advantages*],
//     [*Possible Disadvantages*],
//     [- I'm closer to where you are in terms of experience, perhaps we can relate
//     - I'm close to the code. I write a lot of the code for my research projects and will probably continue to do so.
//     - I'm willing to spend time and effort communicating, mentoring, and teaching],
//     [- #ul[My recommendation counts for less than a recommendation from senior faculty] -- #small[if you're planning on grad school this is an important consideration]
//     - I've mentored undergraduate students before, but I'm generally still new to mentoring.
//     - I can't guarantee any outcomes like publication or conference/journal acceptance from our work together. #small[Senior researchers can't always guarantee this either, but maybe they are better at estimation and project selection.]],
//   )
//   ]

// #table(
//   columns: (50%, auto),
//   stroke: none,
//   [*What's in it for me?*],
//   [*What's in it for you?*],
//   [
//   - *Publication, research progress:* I have several partially-developed research projects. I believe these could become publishable with help from a motivated intern. Without help, they will probably stay stagnant.],
//   [
//   - *(Hopefully) Publication & Coauthorship:* if you contribute significantly to something that reaches publication, you'll be a coauthor on the paper
//   - *Research Experience:* publication or not, you'll get experience with the research process, including:
//     - reading and presenting papers
//     - technical writing
//     - writing code
//     - communicating with other researchers
//   - *Recommendation:* if you do good work, I'll be happy to write you a recommendation for grad school or other opportunities. #small[Again, I'm not a PhD -- this probably will not be nearly as valuable as a recommendation from a senior faculty member.]
//   ],
// )

// #line(length: 100%, stroke: 0.5pt)

// === Who is a good candidate for this internship?

// A good candidate for this internship is someone who:
// - is interested in the topics above
// - is interested in thinking about the #ul[process] of doing research
// - is interested in becoming a more capable technical communicator
// - is interested in becoming a more capable programmer as it relates to research
// - has #ul[*at least 15 hours per week*] to dedicate to this internship
// - is willing to commit to #ul[*at least 1 quarter of work*]

// I'm primarily looking for a student who will be able to advance a project in a self-driven way, with some guidance and structure from me. #small[I do enjoy mentoring less experienced students. However, the main payoff of this internship for me is to have students who can help me advance some research directions I'm excited about.]

// #line()
// === Some Scenarios

// To give you a better sense for how we might work together, here are some hypothetical scenarios. None of these are guaranteed, they're just meant to give you a sense for how you might participate.

// - *Advanced student, ideal outcome:* An advanced student joins a project with some research questions that I defined. Maybe some initial experiments have already been performed. They work on the project mostly independently, with some guidance from me, to run experiments and visualize results. We meet regularly to discuss progress and next steps. After 2-9 months, we write a paper together and submit it to an appropriate venue.

// - *Advanced student, sometimes things don't work as expected:* An advanced student makes considerable progress, but the project doesn't end up being publishable. Perhaps we got scooped, or we obtain inconclusive results. I help the student write a blog post about their experience, perhaps they leverage the experience to apply for other opportunities, or we continue working together on a different project.

// - *Group of advanced and junior students, ideal outcome:* A group of 1-5 students with a range of experience levels join a project with some research questions that I defined. Maybe some initial experiments have already been performed. The advanced students work on the project with some guidance from me, to run experiments and visualize results. The junior students work on the project with more guidance. We meet regularly to discuss progress and next steps. After 2-9 months, we write a paper together and submit it to an appropriate venue. Students who contributed significantly to the project are coauthors on the paper.

// - *Independent Proposal & Mentorship:* An experienced student proposes a project of their own. We work together to refine the project proposal and define some research questions. They work on the project independently, with some advice from me, to run experiments and visualize results. I help if I have time, but the project is mostly theirs. Since the project is further from my own research directions, I rely heavily on the student to take initiative in defining the project and making progress.

// - *Journal Club:* I might pursue this if most of the applicants are less experienced or have limited availability. Instead of running experiments, the students present and discuss papers. I have less material for a recommendation letter than in the above scenarios, but I can still talk about the student's presentation and communication skills. Students gain experience reading and presenting papers, and we all learn something from the discussion.

// #line()

// === The Life Cycle of a Research Project

// #[
// #set text(size: 0.6em)
// #table(
//   columns: (auto, auto, auto, auto, auto),
//   [*Name*],
//   [*Stage 1: Research Questions*],
//   [*Stage 2: Initial Experimentation*],
//   [*Stage 3: Expansion*],
//   [*Stage 4: Finishing*],
//   [*Description*],
//   [Identify a set of research questions],
//   [Run some initial experiments ],
//   [Flesh out the project],
//   [Tie up loose ends, write the paper],
//   [*What does the code look like?*],
//   [Nonexistent],
//   [A single script or notebook],
//   [A code repository with several files],
//   [A code repository with several files],
//   [*What does the writing look like?*],
//   [- Bullet point list of research questions
//   - Maybe some diagrams, slides, or equations
//   - A collection of relevant papers with notes
//   - Maybe a draft of the background & introduction],
//   [- A draft of the introduction and related work
//   - a draft of the methods section],
//   [- Related Work and Intro are outlined at the beginning
//   - Methods & Results are outlined at the beginning, are done at the end
//   ],
//   [A full draft of the paper],
//   [*When do we progress to the next stage?*],
//   [We find a set of research questions that are interesting, timely, and tractable],
//   [We run an initial experiment that shows promising results and good effect sizes],
//   [We have visualizations produced from experimental data that answer one or more research questions],
//   [We have a draft of a paper that answers one or more research questions],
// )
// ]

// === How many students will be accepted?
// #ul[I expect to accept between 1 and 5 students.]

// I don't want to take on more students than I can effectively mentor. If I get a large enough pool with some more advanced students, they might be able to help teach some less experienced students, making it more feasible to take on a larger pool overall.

// = Application

// *Important Note:* if the answer to any of these questions is already obvious in your resume, please feel free to skip it.

// *Personal Details*:
// - Full Name
// - Contact Information (email, phone number)
//   - note that email will be the email used to contact for an interview
// - Educational Background
//   - Major, Year, Expected Graduation Date
//   - Desired internship dates
//   - Weekly availability
//   - How many classes are you taking Fall quarter?
//   - What is your current GPA?
//   - What is your current research experience?

// === AI Theory Background
// - Have you taken any of the following classes or their equivalents?
//   - ECS 170: Intro to AI
//   - ECS 171: Machine Learning

// - Have you taken any courses not in the above list that you think are relevant to this internship? Please list if so

// === Programming Background
// - What is your current programming experience in Python?
//   - zero
//   - heard about it
//   - used it for a small class assignment
//   - used it for a large class assignment
//   - used it for a project larger than a typical course project
// - Please describe the most complex program you've written in Python.
// - If you are more fluent in another language than you are in Python, please describe your experience in that language:
//   - What is your current programming experience in your most fluent language?
//     - heard about it
//     - used it for a small class assignment
//     - used it for a large class assignment
//     - used it for a project larger than a typical course project

// - If you are more fluent in another language than you are in Python, please describe the most complex program you've written in your most fluent language.
// - Have you used an LLM-as-a-service API before? For example, the OpenAI API, Cohere API, or Anthropic API? Please describe your experience.
// - Have you used LLMs locally before? Please describe your experience, including:
//   - models used
//   - the task you were working on
//   - whether prompting or fine-tuning was used

== Take-Home Exercises

The following are two take-home exercises that are meant to:
- Give you a taste of the kind of work you might be doing in this internship
- Give me a sense of your current skills

You are welcome to use any resources you like to complete these exercises,
including:
- Google
- StackOverflow
- ChatGPT, etc.

// I expect that these exercises will take anywhere from 2-6 hours to complete. I know this is a considerable amount of time.

#line(length: 100%, stroke: 0.5pt)
=== Exercise 1: LLM Prompting

==== Before You Start

Please read the instructions for this exercise in full. After you read the
instructions, but before you start your implementation, you're welcome to ask me
high-level questions for clarification. I won't be able to answer questions
about implementation details.

===== Recording Your Work

#underline([* Please record your screen and audio while you complete the steps.*]).
Start the recording when you begin your implementation -- _after_ reading the
instructions and asking and receiving answers to any clarifying questions.
*Video of yourself is not required. Please keep the recording under 3 hours in
duration. It's ok to take a break, you don't have to record all in one sitting.*

I don't expect this to be a polished video, I'm interested in hearing your
thought process as you work through the exercise.

I suggest using Zoom to record your screen and audio, but you're welcome to use
any tool you like.

===== Sharing Your Work
When you're done with this exercise, please:

- Upload your code to a GitHub repository and share the link with me
  (`@g-simmons`)

- Upload your video to Google Drive and share the link with me
  (`gsimmons@ucdavis.edu`).

==== Exercise Instructions

In this exercise, you will use an LLM-as-a-service API to generate text from a
prompt, and explore how the behavior of an LLM varies when prompted with
different "personas".

#let task_counter = counter("task")
#let task(task_name) = block[
  #task_counter.step()
  #text(size: 1em)[*#task_counter.display(). #task_name*]
]

#task([Setup])
- Create an account for a LLM-as-a-service API (OpenAI, Cohere).
- You're free to use any model you like.
- *Please Note:* LLM services cost a small amount of money (fractions of a cent
  per word). Use a moderate sample size and response length to keep costs low.
  - #small[For this exercise you might generate \~500-1000 observations, of perhaps \~100
      tokens each. This would cost \$1.50 or less using chatgpt-3.5-turbo from OpenAI.]
  - #small[Look at the API documentation for your chosen service -- you should be able to
      limit the number of tokens generated per response. I advise you to do so, to
      avoid overspending accidentally.]

#task([Construct and report a "persona space"])
- The "persona space" is the set of personas you will use in your prompt
- For example, a persona might be "a 20-year-old college student" -- this persona
  is one point in a space that you might define with dimensions corresponding to
  age and education level. Other personas in this space might be a 30-year-old
  graduate student, or a 12-year-old middle school student.
- The dimensions of the persona space, and the personas themselves, are up to you.
- Please report your choices in your submission.

#task(
  [Construct and report a template that uses the persona space to generate a prompt],
)
- The template translates each set of variable values comprising a persona into a
  prompt that you will provide to the LLM
- Include the template in your report as a string with some placeholders for the
  personas
- Your template might be something like `You are an {age}-year-old {education
   level}.`

#task([Extend your prompts to elicit some response from the LLM])
- Add text to the template to encourage the LLM to respond in a way that
  facilitates analysis of some response variables
- You can elicit any response you like, but you need to be able to analyze it
- Store your prompts and responses to a file

#task(
  [Analyze the responses from the LLM, show the results in a visualization],
)
- You're free to use any tools you like to analyze the responses and visualize the
  results
- Some examples:
  #[
    #set text(size: 0.8em)
    - Use word counts to compare the frequency of gendered pronouns across different
      personas
    - Use sentiment analysis to compare the sentiment of responses across different
      personas
  ]
- #ul[Only one visualization is expected]
- Interesting visualizations often show:
  - trends (relationships between one variable and another) or
  - comparisons (how one variable differs between two or more groups)

- For example, if you are looking at ice cream preferences by gender, your persona
  space might be `{male, female}`, you might extend your prompt to elicit
  repsonses about ice cream preference, and you might produce a bar chart like
  Figure 1 below:

#figure(
  image("./bar_chart.png", width: 50%),
  caption: [image credit https://statisticsbyjim.com/graphs/bar-charts/],
)

- Write a brief description of your visualization, and what it shows.

- _LLMs don't always cooperate._ Be sure to indicate if you had trouble getting
  the LLM to respond to your prompt, or if you had to try several different
  prompts to get the response you were expecting. If you got any responses that _weren't_ what
  you expected, and you excluded these from your visualization, please indicate
  that as well.

#v(2em)

#pagebreak()
#line(length: 100%, stroke: 0.5pt)
=== Exercise 2: Research Reading & Communication

==== Before You Start
No recording is required for this exercise.

==== Sharing Your Work
After you're done, please upload your summary to a Google Doc and share the link
with me (`gsimmons@ucdavis.edu`).

==== Exercise Instructions
Please read one of the following papers.

#block[
  #set text(size: 0.8em)
  #table(
    columns: (auto, auto),
    stroke: none,
    [1],
    [Argyle, L. P., Busby, E. C., Fulda, N., Gubler, J. R., Rytting, C., & Wingate,
      D. (2023). *Out of One, Many: Using Language Models to Simulate Human Samples.*
      Political Analysis, 1–15. https://doi.org/10.1017/pan.2023.2],
    [2],
    [Aher, G., Arriaga, R. I., & Kalai, A. T. (2023). *Using Large Language Models to
      Simulate Multiple Humans and Replicate Human
      Subject Studies* (arXiv:2208.10264). arXiv.
      https://doi.org/10.48550/arXiv.2208.10264],
    [3],
    [Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., & Hashimoto, T.
      (2023). *Whose Opinions Do Language Models Reflect?* (arXiv:2303.17548). arXiv.
      https://doi.org/10.48550/arXiv.2303.17548],
  )
]

For the paper you chose, write a brief summary (about 1/2 page single spaced) to
answer the following questions:
- What are the main research questions of the paper?
- What are the main findings of the paper?
- What are the methodological strengths and weaknesses of the paper?

  #small[(This generally requires some background information about the domain. I don't
    expect you to have this level of knowledge, so don't worry if you are uncertain
    about your judgements. I'm curious to see what you come up with.)]

- What questions do you have about the paper?

// *Eligibility*:

// - Legal requirements (age, work authorization status)
// - Academic prerequisites (GPA, relevant coursework)

// *Skills and Competencies*:

// - Technical skills relevant to the research project(s)
//   - LLM Prompting & Fine-Tuning
//   - Research Reading & Communication
// - Soft skills (communication, teamwork, problem-solving)
//   -

// *Experience*:

// - Previous internship/research experiences
// - Relevant project portfolios

// *Presentation*:

// 1. *Project Descriptions*:

//   - Clear and concise descriptions of each research project
//   - Outline of the roles and responsibilities of the interns

// 2. *Visual Appeal*:

//   - A well-designed form with a logical flow
//   - Use of branding elements (logos, color schemes)

// ### 4. *Personalization*:

// 1. *Preference Indication*:

//   - Allowing applicants to indicate their project preferences
//   - Optional fields for additional information

// 2. *Personal Statements*:

//   - Short essays or statements to understand the motivations and aspirations of the applicants

// ### 5. *Feedback and Support*:

// 1. *Confirmation*:

//   - Sending a confirmation message upon successful submission
//   - Providing a contact for further inquiries

// 2. *Feedback Loop*:

//   - Option to request feedback on the application
//   - Informing applicants of the selection timeline

// ### 6. *Legal and Ethical Considerations*:

// 1. *Privacy*:

//   - Clearly stating the privacy policy concerning the handling of personal data
//   - Obtaining consent for data processing

// 2. *Equality and Diversity*:

//   - Encouraging applications from diverse backgrounds
//   - Avoiding discriminatory questions

// I don't want to take on more students than I can effectively mentor, and there are some dependencies between students in the applicant pool. If I get a large enough pool with some more advanced students, they might be able to help teach some less experienced students, making it more feasible to take on a larger pool overall.

// I'll likely contact you for an interview soon!
// If you don't hear back by X date, that unfortunately means that I'm not moving forward with the application for now.

// Maybe a question to mark subsequent quarters that you would be interested in working.

// A policy on coauthorship with some examples:
