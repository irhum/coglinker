from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def hallucination_chain(api_key):
    """Chain for hallucinating a passage from an abstract and a rough thought.
    
    This chain produces takes two inputs: an abstract and a "rough thought."
    The outputs are a refined question (generated from the rough thought) and a 
    fabricated passage (intended to be used as a vector search query).
    """

    chat = ChatOpenAI(temperature=0.1, openai_api_key=api_key)
    template = """We are playing a game, where I am an expert, and your goal is to generate a 100 word passage so realistic I cannot tell whether it is generated or from a real paper.
    I'm giving you the abstract for the paper I am currently reading. I am also giving you my "rough thought" to help you guide your generation.
    Once you decode my "rough thought", fabricate a passage with precise technical details (e.g. 18.5x increase, $2x^5 \sin(x)$, HeLa cells).
    Keep both the decoded thought and passage short."""
    system_message_prompt = SystemMessage(content=template)

    # One-shot example with text from from https://arxiv.org/abs/2210.15097
    example_human1 = HumanMessage(
        content="Abstract: Likelihood, although useful as a training loss, is a poor search objective for guiding open-ended generation from language models (LMs). Existing generation algorithms must avoid both unlikely strings, which are incoherent, and highly likely ones, which are short and repetitive. We propose contrastive decoding (CD), a more reliable search objective that returns the difference between likelihood under a large LM (called the expert, e.g. OPT-13b) and a small LM (called the amateur, e.g. OPT-125m). CD is inspired by the fact that the failures of larger LMs are even more prevalent in smaller LMs, and that this difference signals exactly which texts should be preferred. CD requires zero training, and produces higher quality text than decoding from the larger LM alone. It also generalizes across model types (OPT and GPT2) and significantly outperforms four strong decoding algorithms in automatic and human evaluations.\n\nRough thought: objective to maximize\n\nNow, fabricate."
    )
    example_ai1 = AIMessage(
        content="""Decoded thought: What is the formula for the objective maximized?\n\nPassage: contrastive decoding searches for text that maximizes the contrastive objective $\mathcal {L}_{\text{CD}} = \log p_\textsc {exp}(\textsf {x$ cont $}\mid \textsf {x$ pre $}) - \log p_\textsc {ama}(\textsf {x$ cont $}\mid \textsf {x$ pre $})$ , subject to constraints that $\textsf {x$ cont $}$ should be plausible (i.e., achieve sufficiently high probability under the expert LM)."""
    )
    human_template = """Abstract:{abstract}\n\nWrite a scientific paper passage to answer the question: {query}\n\nNow, fabricate."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            example_human1,
            example_ai1,
            human_message_prompt,
        ]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    return chain


def synthesis_chain(api_key):
    """Chain for synthesizing an answer from an abstract, retrieved passaged and a question."""
    chat = ChatOpenAI(temperature=0.1, openai_api_key=api_key)

    template = """You are taking a technical writing assessment, where you are provided the following:
        1. The abstract for a research paper.
        2. A question to be answered about the research paper.
        3. Full length passages from the paper that are relevant to answering the question.

    To achieve a high score, you must:
        1. Produce a factual answer to the question asked. Your answer should be in formatted Markdown.
        2. Seamlessly integrate information from the passages provided. Such information may include equations, quotes and technical acronyms.
        3. Ensure your answer is concise and clear. Do not add unnecessary detail beyond what is strictly needed, as this may confuse your examiner.
        """

    system_message_prompt = SystemMessage(content=template)
    human_message_prompt1 = HumanMessagePromptTemplate.from_template(
        """Paper Abstract: {abstract}
        Question: {question}
        Relevant passages: {retr_passages}
        
        Using the instructions provided, answer the question."""
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            human_message_prompt1,
        ]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    return chain
