# `raggy` - a scraping and querying library
![hero](assets/logos/raggy.png)

#### shhhhhh... I'm just here for the code
```bash
pip install raggy
```
see the [tutorial](welcome/tutorial.md) for a quick start

## What is this _RAG_ thing people keep talking about?
R.A.G. stands for _Retrieval Augmented Generation_. Enough hot air has been blown about it, so just read [this](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) if you're not sure what it is.

## Why is it useful?
Large language models are trained on discrete data sets, at points in time. They do not update their knowledge sources inherently. 
Humans want LLMs to know about _their_ data, right now.

## What does `raggy` do?

### Non-technical
- efficiently scrape data from the web into rich documents
- throw these documents into a place where they're grouped by similarity and labelled with metadata
- retrieve documents from this place on command, based on similarity to a query or specific metadata (e.g. dates, authors, etc.)

### More technical
- coerce arbitrary data formats into a list of `Document` objects, possibly enriched with metadata
- upsert these `Document` objects into a vectorstore like `Chroma` or `Turbopuffer`
- expose human / LLM-friendly query interfaces to these vectorstores that allow metadata filtering