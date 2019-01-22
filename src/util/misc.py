def extract_sentence(sentence_id, wiki_data):
    """
    Given a corpus dictionary item wiki_data, extract sentence with sentence_id from available lines.

    Args:
        sentence_id (int): number of sentence to extract
        wiki_data (dict): Wikipedia-dump dictionary object containing the content of 1 page
    Raises:
        RuntimeError: On attempts to retrieve a sentence from an empty dict, 
                      or when a sentence_id exceeds the dict content
    Returns:
        A string containing the sentence specified by sentence_id.
    """
    if not wiki_data['id']:
        raise RuntimeError("extract_sentence() received call to read {} from empty wiki_data".format(sentence_id))
    
    # determine amount of digits in sentence_id
    digits = len(str(abs(sentence_id)))

    # find "<num>\t" in the string to determine sentence starting index, offset by 1 + amount of digits sentence_id has
    sentence_start = wiki_data['lines'].find("{}\t".format(sentence_id)) + (digits + 1)
    # find end of sentence, denoted by the first ".\t" after the start index of the sentence, 1-offset to include period
    sentence_end = wiki_data['lines'].find(".\t", sentence_start) + 1
    print(sentence_start, sentence_end)
    if sentence_start == -1 or sentence_end == -1 or (sentence_id > 1 and sentence_start == digits):
        # -1 translates to not found
        # however, doesn't seem to apply in this context, find() has trouble with partial matches and erroneously
        # returns (amount of digits, end of first line of text) as start/end. So whenever sentence_id is larger than 1 
        # and sentence_start is one of those suspicious values, we raise an error. Works reliably so far.
        raise RuntimeError("sentence_id {} not found for wiki_id {}".format(sentence_id, wiki_data['id']))
    else:
        return wiki_data['lines'][sentence_start:sentence_end]