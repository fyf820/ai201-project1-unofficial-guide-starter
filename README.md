# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->
San Francisco restaurant recommendations. This konwledge valuable because it contains the wait time/difficulty of appointment, quality, and recommended food for each restaurant. The official channels only shows menus, location, and business hours. They rarely contains cumstomer experience and food quility.
---


## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 |Eater SF Article|The 38 Best Restaurants in San Francisco|https://sf.eater.com/maps/best-restaurants-san-francisco-38|
| 2 |Reddit Thread|Restaurant Recommendation and Omission List in SF|https://www.reddit.com/r/AskSF/comments/1j178op/san_francisco_restaurant_list_recommendations_and/|
| 3 |Reddit Thread|Currently the best restaurants in San Francisco|https://www.reddit.com/r/AskSF/comments/1rad0pt/what_is_currently_the_best_restaurants_in_san/|
| 4 |Article|Top 100 Restaurants in the Bay Area in 2025|https://www.sfchronicle.com/projects/2025/top-100-best-restaurants-san-francisco-bay-area/|
| 5 |Quora Question|Recommend restaurant for someone visiting San Francisco|https://www.quora.com/What-restaurant-would-you-recommend-to-someone-visiting-San-Francisco|
| 6 |Quora|Must try San Francisco restaurants|https://www.quora.com/What-are-some-must-try-San-Francisco-restaurants|
| 7 |Article|San Francisco’s Top 10 Restaurants To Visit|https://www.jsfashionista.com/san-franciscos-top-10-restaurants/|
| 8 |Yelp Restaurant Page|Most reviewed Restaurants "Zushi Puzzle" in SF|https://www.yelp.com/biz/zushi-puzzle-san-francisco-2?osq=Restaurants|
| 9 |Reddit Thread|The Most "San Francisco" Restaurants in SF|https://www.reddit.com/r/AskSF/comments/1le4ps5/the_most_san_francisco_restaurants_in_sf/|
| 10 |Michelin Guide of San Francisco Restaurants|Michelin rated and recommended restaurants in San Francisco|https://guide.michelin.com/us/en/california/san-francisco/restaurants|

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**
I'll use paragraph chunks, it will be 300-400 token for each chunk. 
**Overlap:**
20-50 tokens 
**Why these choices fit your documents:**
Because my resources are mostly short reviews or short answers, even articles are one paragraphs for each restaurant. Paragraph chunks can keep the meaning better.
**Final chunk count:**
156
---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
all-MiniLM-L6-v2
**Production tradeoff reflection:**
As I am using free api key, I'll use bigger and heavier model for a accurate answer, but the tradeoff is that there might be some latency. As I mentioned before, I want to keep paragraph chunk to avoid lose information, so the model will have a longer context length. Most of the materials are common food terms, there is few food specific terms.
---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
You are a knowledgeable guide to San Francisco restaurants. \
Answer the question using only the information in the provided documents.

Rules:
1. Use ONLY information found in the CONTEXT documents. Do not rely on outside knowledge.
2. If the documents don't contain enough information to answer, say exactly: \
"I don't have enough information on that."
3. Name specific restaurants and quote concrete details (dishes, prices, \
atmosphere, wait times) when they appear in the context.
4. After each claim, cite the source filename(s) it came from in brackets, \
e.g. [8-yelp-zushi-puzzle.txt].
5. Be concise and do not invent restaurants, reviews, or facts.
**How source attribution is surfaced in the response:**
It will be cited inline with in brackets in the responce and append a Sources list to the response
---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 |What do people say about the food quality and atmosphere at Zushi Puzzle?|Reviews generally describe it as a high-quality sushi spot with fresh fish, creative rolls, and a modern but casual atmosphere; often praised for taste but sometimes noted as pricey.|People have mixed reviews about the food quality and atmosphere at Zushi Puzzle.|Partially relevant|Accurate|
| 2 |What does the Michelin guide say about the best restaurant in SF for a celebration meal?|The Michelin-rated restaurant name plus mention of celebration|The documents provided do not contain information about the Michelin guide's ranking of the best restaurant in SF for a celebration meal. Listed some restaurant with stars but no rank.|Partially relevant|Partially accurate |
| 3 |Which restaurant is facing parking or transit challenges?|The restaurant name plus a short statement about parking/transit issues.|I don't have enough information on that.|Off-target|Inaccurate|
| 4 |What restaurants would you recommend for a group dinner that offers good value for money?|The restaurant name with supporting detail about affordability or group-friendly atmosphere.|Recommend some restaurant for a group dinner that offers good value for money and list resonales|Relevant| Accurate|
| 5 |Which SF restaurant is recommended for a first-time visitor who wants classic local food?|A specific restaurant plus a short reason|Recommend some restaurant for a first-time visitor who wants classic local food and list resonales|Relevant|Accurate|

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**
Which restaurant is facing parking or transit challenges?

**What the system returned:**
I don't have enough information on that.

**Root cause (tied to a specific pipeline stage):**
The failure happens at the chunking and embedding/retrieval stages, for two compounding reasons. Signal dilution during chunking, The corpus does contain a relevant passage, but chunk that is otherwise about the pizza, the neighborhood, and the menu. When that whole chunk is compressed into one embedding vector, the parking sentence is averaged out by the surrounding food content, so the chunk never ranks near a parking-focused query. Another root cause is lexical ambiguity. The documents are full of location uses of park that are semantically unrelated to car parking, which pulls retrieval toward the wrong sense of the word.

**What you would change to fix it:**
 I can add hybrid search, add a lexical search alongside the embedding search and merge results

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**
The chunking and retrieval approach is built very seamlessly and works well. It saves much time and I can put more time and effects on other part.

**One way your implementation diverged from the spec, and why:**
I changed 2 test questions in my Evaluation plan because they have a large distance. One of the question has more than 0.8 distance but lower to about 0.5 after I changed the question. Also, instead of asking restaurant, it just ask for the information in db, it is not a good question.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:* I gave Copilot the URL lists of documents and ask it to fetch contents, save them as text in the documents file.  For those cannot acess, just tell me and create a new text file for them, I will paste them.
- *What it produced:* 10 documents but it only stored a very small parts of the fetched files. Others cannot access just left the URLs.
- *What I changed or overrode:* Delete all the contents and manually collect documents

**Instance 2**

- *What I gave the AI:* Remove HTML tags, navigation menus, cookie banners, ads, footers, repeated site headers, "Read more" links, share buttons, comment counts, and any boilerplate that appears on every page.Just keep the actual review text, opinions, ratings, descriptions, and any context needed to understand the content (e.g., the professor's name or course number in a review).
- *What it produced:* Remove some ads but still contains ads. Some repeated contents, user avatar placeholder, notification of deleted posts.
- *What I changed or overrode:* I override the rule and give more specific instruction to clean the documents.


## Hybrid Search
Implement hybrid search a lexical search alongside the embedding search and merge results. The Q3 parking now surfaces the answer chunk semantic missed. Q1/Q4/Q5 was reordered within the same good sources.
## Video Link
https://drive.google.com/file/d/11q66Un5MVg4I3M3HKng6ALOcHPhU9wmg/view?usp=sharing