# Coglinker
What if you could ask an arXiv preprint questions directly?

This is a (really) early set of experiments on how LLMs can be used to build tools that help people understand research papers faster (as opposed to just directly reading a PDF). 

## Demo
![Coglinker Demo](LINK)

Here's a Colab version of this you can have up and running in ~2 minutes: [Coglinker Demo](LINK)

## How it works
In its current implementation, coglinker:

1. Directly downloads the source LaTeX for a requested arXiv paper.
2. Parses the LaTeX, and processes it into chunks.
3. Given a question, retrieves relevant "chunks" and adds it to an LLM's (current ChatGPT) context.
4. Generates an answer to the question using the LLM.

## Open Questions
* How do we build tooling that helps people maximize their [transfer of learning](https://en.wikipedia.org/wiki/Transfer_of_learning) when they read (and engage with) research?
