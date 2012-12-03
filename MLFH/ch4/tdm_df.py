def tdm_df(doclist, stopwords = [], remove_punctuation = True,
           remove_digits = True, sparse_df = True):
    '''
        
    Create a term-document matrix from a list of e-mails.

    Uses the TermDocumentMatrix function in the `textmining` module.
    But, pre-processes the documents to remove digits and punctuation,
    and post-processes to remove stopwords, to match the functionality
    of R's `tm` package.

    NB: This is not particularly memory efficient and you can get memory
    errors with an especially long list of documents.

    Returns a (by default, sparse) DataFrame. Each column is a term,
    each row is a document.
    '''
    import numpy as np
    import textmining as txtm
    import pandas as pd
    import string

    # Some (at least to me) unavoidable type-checking.
    # If you only pass one document (string) to the doclist parameter,
    # the for-loop below will iterate over the letters in the string
    # instead of strings in a list. This coerces the doclist parameter
    # to be a list, even if it's only one document.
    if isinstance(doclist, basestring):
        doclist = [doclist]
 
    # Create the TDM from the list of documents.
    tdm = txtm.TermDocumentMatrix()
    
    for doc in doclist:
        if remove_punctuation == True:
            doc = doc.translate(None, string.punctuation.translate(None, '"'))
        if remove_digits == True:
            doc = doc.translate(None, string.digits)

        tdm.add_doc(doc)

    # Push the TDM data to a list of lists,
    # then make that an ndarray, which then
    # becomes a DataFrame.
    tdm_rows = []
    for row in tdm.rows(cutoff = 1):
        tdm_rows.append(row)

    tdm_array = np.array(tdm_rows[1:])
    tdm_terms = tdm_rows[0]
    df = pd.DataFrame(tdm_array, columns = tdm_terms)

    # Remove stopwords from the dataset, manually.
    # TermDocumentMatrix does not do this for us.
    if remove_punctuation:
        stopwords = [w.translate(None, string.punctuation.translate(None, '"'))
                     for w in stopwords]
    if len(stopwords) > 0:
        for col in df:
            if col in stopwords:
                del df[col]

    if sparse_df == True:
        df.to_sparse(fill_value = 0)

    return df
