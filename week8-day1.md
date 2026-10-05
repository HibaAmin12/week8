# Week 08 · Mon · Learn
## RAG Basics — Chunking, Embeddings, Vector Databases, and Retrieval Evaluation

### Learning objectives
By the end of today, you should be able to:
- Explain what problem RAG actually solves, and why "just give the model more context" isn't automatically a substitute for real retrieval.
- Draw the full RAG pipeline from raw document to grounded answer, naming what happens at ingestion time versus query time.
- Explain the real tradeoff in chunk size and chunk overlap, and choose defensible starting values instead of a guess.
- Explain what a text embedding actually is, and why cosine similarity is the right tool for comparing two of them.
- Explain what problem a vector database solves that a plain list of embeddings and a for-loop doesn't, at real scale.
- Distinguish semantic search, keyword search, and hybrid search, and explain a concrete case where embeddings alone fail.
- Explain the two-stage retrieve-then-rerank pattern, and why a cross-encoder isn't used for the first retrieval pass despite being more accurate.
- Apply Week 6's precision/recall vocabulary to retrieval quality specifically — precision@k and recall@k on retrieved chunks, not classification labels.
- Explain the "lost in the middle" finding and why it's a real, measured reason precise retrieval still matters even as context windows grow.

### Framing
Weeks 6 and 7 built two separate skills: evaluating a model rigorously (baselines, precision/recall, error analysis) and calling a model well (prompting, tool use, structured output, local and hosted). RAG is where those two skills combine with one new one — search — to solve a problem neither prompting nor fine-tuning solves cleanly: getting a model to answer correctly using information it was never trained on, and wasn't told directly in the conversation. Today is entirely foundations, the same shape as every other Monday this program has run: no production build yet, but every concept here gets used directly starting tomorrow, and this week's roadmap explicitly expects Week 6's evaluation vocabulary to already be second nature — retrieval evaluation is a recap applying it to a new object, not a first exposure to precision and recall.

### Lesson

**1. What RAG actually solves**
A model's knowledge is frozen at training time and limited to what's in its context window on any given request — it doesn't know about your private documents, your company's internal wiki, or anything published after its training cutoff, and it can't be retrained per document you want it to know about. **Retrieval-Augmented Generation** solves this by keeping your actual data outside the model entirely, searching it for the pieces relevant to a specific question at the moment the question is asked, and inserting just those pieces into the prompt before generating an answer. The model isn't "taught" your data — it's handed the relevant slice of it, fresh, every single request.

**2. The full pipeline, end to end**
Two distinct phases, worth keeping separate in your head: **ingestion time** (done once, or whenever your source documents change) — load documents, split them into chunks, compute an embedding for each chunk, store the chunk text and its embedding in a vector database. **Query time** (done on every user request) — embed the incoming question with the same embedding model, search the vector database for the chunks whose embeddings are most similar to the question's embedding, optionally re-rank those candidates, then build a prompt containing the retrieved text plus the original question and generate an answer grounded in it. Everything in today's lesson is a piece of one of these two phases.

**3. Chunking — why, and the real tradeoff**
You can't embed and retrieve a whole document as one unit — a long document covers many topics, so its single embedding would be a vague average of all of them, matching poorly against a question about any one specific part. **Chunking** splits documents into smaller pieces before embedding each one separately. Fixed-size chunking (by token or character count) is simple but can cut a sentence or an idea in half at an arbitrary boundary; **overlap** — repeating a portion of one chunk's end at the start of the next — mitigates that by keeping cut-off context available in a neighboring chunk. A practical, well-supported starting point: roughly 512 tokens per chunk with 10–20% overlap, tuned from there based on actual retrieval results, not guessed once and left alone. The real tradeoff to hold in mind: chunks too large dilute relevance (a match on one sentence drags in a lot of unrelated surrounding text, and wastes context-window space) and chunks too small lose the surrounding context that made the fragment meaningful in the first place.

**4. Embeddings — meaning as geometry**
An **embedding** is a vector — a fixed-length list of numbers — produced by a model trained so that texts with similar meaning end up with similar vectors, geometrically. This is what makes **semantic search** possible: instead of matching exact words, you're comparing meaning, positioned in a high-dimensional space. **Cosine similarity** — the cosine of the angle between two vectors — is the standard comparison metric here specifically because it's magnitude-invariant: it measures whether two vectors point in the same *direction*, regardless of length, which matters because embedding vector length isn't a meaningful signal the way its direction is. Two texts with a cosine similarity near 1 are semantically close; near 0, unrelated.

**5. Vector databases — the problem of scale**
Computing cosine similarity between a query and one stored embedding is trivial; computing it against a few million stored embeddings, one at a time, in a loop, on every single query, is not fast enough for a real application. A **vector database** solves this with approximate nearest-neighbor indexing (Qdrant, for instance, uses an algorithm called HNSW) — trading a small, tunable amount of retrieval accuracy for a massive speedup, so a similarity search over millions of vectors returns in milliseconds instead of seconds. Dedicated vector databases (Qdrant, Pinecone, Chroma) are purpose-built for this; **pgvector** instead adds vector search as an extension to a Postgres database you may already be running — the practical choice between them is often less about raw capability and more about whether you want one more piece of infrastructure to operate, or you'd rather extend a database you already have.

**6. Semantic search isn't the whole answer — hybrid search**
Embeddings are excellent at matching meaning and terrible at matching a specific token exactly — a product code, an error message, a person's name, or an acronym can be semantically "close to" many things while an embedding model has no special reason to treat an exact character match as important. Keyword search (commonly BM25) does the opposite well: exact and near-exact term matches, no sense of meaning beyond the literal words. **Hybrid search** runs both in parallel and combines the results, which is why it's the common production default rather than either alone — you get semantic matching for conceptual questions and reliable exact-term recall for the cases embeddings systematically miss.

**7. Re-ranking — retrieve broadly, then rank precisely**
Two different kinds of embedding-based models matter here. A **bi-encoder** (what you use for the initial retrieval) encodes the query and every document independently, ahead of time for documents, so comparison at query time is just a fast vector lookup — but because query and document never interact during encoding, its relevance judgments are comparatively loose. A **cross-encoder** encodes the query and one candidate document *together*, letting it directly attend across both — far more accurate at judging true relevance, but too slow to run against your entire corpus on every query. The standard pattern: use the fast bi-encoder to retrieve a broad candidate set (top 50, say), then use the slower, more accurate cross-encoder to re-rank just those candidates down to the final top few actually sent to the model. This two-stage pattern is why re-ranking exists as a separate step rather than just "using the better model everywhere" — production RAG systems commonly see a real, measurable relevance improvement from this second pass, for well under 200ms of added latency.

**8. Retrieval evaluation — precision and recall, on chunks**
Week 6 built precision and recall around a classification label being right or wrong. Retrieval evaluation asks the same underlying question of a *retrieved set*: of the chunks your pipeline retrieved for a query, **precision@k** asks what fraction of the top-k retrieved chunks were actually relevant; **recall@k** asks what fraction of *all* the relevant chunks in your corpus made it into the top-k retrieved set at all. A RAG system with excellent generation but poor recall@k is structurally broken in a way better prompting can't fix — the model can only answer using what retrieval handed it, so if the relevant chunk was never retrieved, no amount of prompt engineering recovers it.

**9. Long context vs. real retrieval — the "lost in the middle" problem**
As context windows have grown, "just put everything in the prompt and skip retrieval" is a real, tempting option for smaller corpora — but two things push back on it even when it would technically fit. First, cost: every token in the prompt is billed and latency-relevant on every single request, whether or not it's relevant to that specific question. Second, and more surprising: controlled research (Liu et al., "Lost in the Middle") found that models use long contexts unevenly — performance on finding and using relevant information is measurably strongest when that information sits near the start or end of the context, and measurably degrades when it's buried in the middle, forming a U-shaped curve, even in models explicitly built for long-context input. Precise retrieval — handing the model a short, high-relevance set of chunks instead of everything — sidesteps this failure mode entirely rather than hoping the model finds the needle wherever it happens to land in a much longer haystack.

### Resources
- [Qdrant — Documentation](https://qdrant.tech/documentation/)
- [OpenAI — Vector embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [Weaviate — Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag)
- [Towards Data Science — Hybrid Search and Re-Ranking in Production RAG](https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/)
- [Liu et al. — Lost in the Middle: How Language Models Use Long Contexts (arXiv)](https://arxiv.org/pdf/2307.03172)

### Kata set

**Part A — chunking and embeddings, by hand first**
1. **Chunk a real document two ways.** Take any real multi-paragraph document (a README, an article). Split it with fixed-size chunking (pick ~300–500 tokens, roughly, by word count as a stand-in) with no overlap, then again with ~15% overlap. Find one real spot where the no-overlap version cuts an idea awkwardly in half, and confirm the overlapping version keeps it intact in at least one chunk.
2. **Compute cosine similarity yourself.** Using any embedding model you have access to (a hosted API's embeddings endpoint, or a small local model), embed three short sentences — two about similar topics, one unrelated — and compute the pairwise cosine similarity yourself with plain NumPy (`np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`). Confirm the two related sentences score higher than either does against the unrelated one.

**Part B — a real vector database, end to end**
3. **Set up Qdrant locally.** `pip install qdrant-client` — use its in-memory mode for today (no server to run separately). Create a collection, embed 8–10 real sentences on a mix of topics, and store them with their text as payload alongside the vector.
4. **Query it and verify real retrieval.** Embed a query sentence related to only one of your topics, search the collection, and confirm the top result is actually the sentence you'd expect a human to pick — not just that the code runs without error.

**Part C — hybrid search and evaluation, made concrete**
5. **Break semantic search on purpose.** Add one sentence to your Part B collection containing a specific made-up code or ID (e.g., "Ticket REF-4471 was resolved by rotating the API key"). Query for that exact ID string. Note whether pure vector similarity search actually surfaces it reliably — this is the concrete case hybrid search exists to fix.
6. **Measure precision@k and recall@k on your own tiny corpus.** For 3 queries against your Part B/C collection where you already know which sentence(s) are actually relevant, retrieve the top 3 results for each and compute precision@3 and recall@3 by hand. State one honest finding from your own numbers, not a hypothetical one.

### Today's tasks
- [ ] Document chunked two ways (no overlap vs. ~15% overlap), one real awkward cut identified and shown fixed
- [ ] Cosine similarity computed by hand on 3 real embedded sentences, related pair scoring higher than the unrelated pair
- [ ] Qdrant collection created locally with 8–10 real embedded sentences stored
- [ ] A real query run against the collection, top result verified sensible by inspection
- [ ] Exact-ID query tested against pure vector search, result honestly reported (worked or didn't)
- [ ] Precision@3 and recall@3 computed by hand on 3 real queries against your own small corpus


