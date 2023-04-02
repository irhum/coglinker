def process_passages(raw_passages):
    """Process raw passages from json into a list of strings.
    
    Args: 
        raw_passages: List of dicts with keys "text", "section", and "eq_spans".
    """
    passages = []
    prev_section = None

    # Loop through raw passages
    for raw_passage in raw_passages:
        # These are strings with "reference IDs" for equations/tables/figs
        passage = raw_passage["text"]
        section = raw_passage["section"]

        # replace equation ref ids with actual latex
        for span in raw_passage["eq_spans"]:
            passage = passage.replace(span["ref_id"], f"${span['latex']}$")

        # Merge passages if they are from the same section OR if the passage is short
        if section == prev_section or (len(passage) < 300 and len(passages[-1]) < 1800):
            passages[-1] = passages[-1] + passage
        else:
            passages.append(passage)

        # Update previous section
        prev_section = section

    proc_passages = []

    # Split long passages into chunks
    for passage in passages:
        if len(passage) < 1800:
            proc_passages.append(passage)
        else:
            MAX_CHUNK = 1200
            passage_chunks = [
                passage[i : i + MAX_CHUNK] for i in range(len(passage), MAX_CHUNK)
            ]
            proc_passages.extend(passage_chunks)

    return proc_passages
