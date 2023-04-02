import json
import itertools

import numpy as np
import streamlit as st
import pandas as pd

import os
from urllib import request
from langchain import embeddings
from coglinker import chains, preprocess
from doc2json.tex2json import process_tex

# Set the page layout to wide
st.set_page_config(layout="wide")

cols = st.columns([3, 1])

with cols[0]:
    query = st.text_input("What's your question?")

with cols[1]:
    st.markdown("Papers to use for retrieval")
    center_id = st.text_input("arXiv ID for center paper", "2203.11618")
    df = pd.DataFrame([{"arXiv ID": f"{center_id}", "activated": True}])
    edited_df = st.experimental_data_editor(df, num_rows="dynamic")

# Setup chains
api_key = os.environ["OPENAI_API_KEY"]
embed = embeddings.OpenAIEmbeddings(openai_api_key=api_key)
gen_chain = chains.hallucination_chain(api_key)
ans_chain = chains.synthesis_chain(api_key)


# Download JSON for paper
@st.cache_data(persist=True)
def download_json(idx):
    _ = request.urlretrieve(f"https://arxiv.org/e-print/{idx}", f"{idx}.gz")
    process_tex.process_tex_file(f"{idx}.gz", output_dir="output")
    with open(f"output/{idx}.json") as f:
        data = json.load(f)
    return data

# Process JSON for paper
@st.cache_data(persist=True)
def process_json(data):
    abstract = "".join([texts["text"] for texts in data["latex_parse"]["abstract"]])
    passages = preprocess.process_passages(data["latex_parse"]["body_text"])
    embeds_resp = np.array(embed.embed_documents(passages))
    return abstract, passages, np.array(embeds_resp)

# Download and process paper
def load_paper(idx):
    data = download_json(idx)
    return process_json(data)

# If query is not empty, run the pipeline
if query != "":
    # Load abstract for center paper
    abstract, _, _ = load_paper(center_id)

    # Load passages and embeddings for all papers
    all_passages, all_embeds = {}, {}
    for idx in edited_df["arXiv ID"]:
        _, passages, embeds = load_paper(idx)
        all_passages[idx] = passages
        all_embeds[idx] = embeds

    # Concatenate passages and embeddings for activated papers
    activated = {
        idx: activated
        for (idx, activated) in zip(edited_df["arXiv ID"], edited_df["activated"])
    }
    passages = list(
        itertools.chain(
            *[value for (idx, value) in all_passages.items() if activated[idx]]
        )
    )
    embeds = np.concatenate(
        [value for (idx, value) in all_embeds.items() if activated[idx]], axis=0
    )

    # Create hallucinated passage
    hallucinated = gen_chain.run(abstract=abstract, query=query)
    o = [s.split(":", 1)[1] for s in hallucinated.split("\n\n", 1)]
    # Extract refined question and generated passage
    refined_q, gen_passage = o[0], o[1]
    # Embed generated passage
    gen_passages_emb = embed.embed_query(gen_passage)
    # Retrieve top 3 passages closest to generated passage
    retr_passages = [passages[i] for i in np.argsort(-embeds @ gen_passages_emb)][:3]
    retr_passages_str = "\n".join(
        [f"Passage {i} - <<{passage}>>" for (i, passage) in enumerate(retr_passages)]
    )

    # Display refined question
    with cols[0]:
        st.markdown(f"### Expanded Question:\n\n{refined_q}")

    # Generate and display answer
    st.markdown(
        f"### Answer:\n\n{ans_chain.run(abstract=abstract, question=refined_q, retr_passages=retr_passages_str)}"
    )
    
    # Display retrieved passages
    with st.expander("See retrieved passages"):
        cols = st.columns(3)

        for col, passage in zip(cols, retr_passages):
            with col:
                st.markdown(passage)
