''''
-------------------------------------------------------------------------------
Filename   : ch3.ipynb
Date       : 2012-06-17
Author     : C. Vogel
Purpose    : Replicate the naive Bayes e-mail classifier in Chapter 3 of
           : _Machine Learning for Hackers_.
Input Data : e-mail files, split into spam and ham (non-spam) folders are available
           : at the book's github repository at https://github.com/johnmyleswhite/
           : ML_for_Hackers.git. This also uses r_stopwords.csv, a text file
           : containing a list of stopwords used by R's tm package. This is used
           : to facilitate comparability with the results of the R analysis.
Libraries  : Numpy 1.6.1, Pandas 0.7.3, NLTK 2.0.1, textmining
-------------------------------------------------------------------------------

This notebook is a Python port of the R code in Chapter 3 of _Machine Learning
for Hackers_ by D. Conway and J.M. White.

E-mail files, split into folders classified as spam or ham (non-spam) should be located
in a /data/ subfolder of the working directory. See the paths defined just after the import
statements below to see what directory structure this script requires. Copying complete
data folder from the book's github repository should be sufficient.

For a detailed description of the analysis and the process of porting it
to Python, see: slendrmeans.wordpress.com/will-it-python.
'''

import os
import math
import string
import nltk
from nltk.corpus import stopwords
import numpy as np
import textmining as txtm
from pandas import *

# Directories with e-mail data
# The spam and ham files are broken into multiple
# directories so as to separate training and evaluation data
data_path = os.path.abspath(os.path.join('.', 'data'))
spam_path = os.path.join(data_path, 'spam')
spam2_path = os.path.join(data_path, 'spam_2')
easyham_path = os.path.join(data_path, 'easy_ham')
easyham2_path = os.path.join(data_path, 'easy_ham_2')
hardham_path = os.path.join(data_path, 'hard_ham')
hardham2_path = os.path.join(data_path, 'hard_ham_2')

def get_msg(path):
    '''
    Read in the `message` portion of an e-mail, given
    its file path. The `message` text begins after the first
    blank line; above is header information.

    Returns a string.
    '''
    with open(path, 'rU') as con:
        msg = con.readlines()
        first_blank_index = msg.index('\n')
        msg = msg[(first_blank_index + 1): ]
        return ''.join(msg)

def get_msgdir(path):
    '''
    Read all messages from files in a directory into
    a list where each item is the text of a message.

    Simply gets a list of e-mail files in a directory,
    and iterates `get_msg()` over them.

    Returns a list of strings.
    '''
    filelist = os.listdir(path)
    filelist = filter(lambda x: x != 'cmds', filelist)
    all_msgs =[get_msg(os.path.join(path, f)) for f in filelist]
    return all_msgs

# Get lists containing messages of each type.
all_spam = get_msgdir(spam_path)
all_easyham = get_msgdir(easyham_path)
all_easyham = all_easyham[:500]
all_hardham = get_msgdir(hardham_path)

# Get stopwords.
# NLTK stopwords
sw = stopwords.words('english')
# Stopwords exported from the 'tm' library in R.
rsw = read_csv('r_stopwords.csv')['x'].values.tolist()

def tdm_df(doclist, stopwords = [], remove_punctuation = True,
           remove_digits = True, sparse_df = False):
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
    df = DataFrame(tdm_array, columns = tdm_terms)

    # Remove stopwords from the dataset, manually.
    # TermDocumentMatrix does not do this for us.
    if len(stopwords) > 0:
        for col in df:
            if col in stopwords:
                del df[col]

    if sparse_df == True:
        df.to_sparse(fill_value = 0)

    return df

spam_tdm = tdm_df(all_spam, stopwords = rsw, sparse_df = True)

def make_term_df(tdm):
    '''
    Create a DataFrame that gives statistics for each term in a
    Term Document Matrix.

    `frequency` is how often the term occurs across all documents.
    `density` is frequency normalized by the sum of all terms' frequencies.
    `occurrence` is the percent of documents that a term appears in.

    Returns a DataFrame, with an index of terms from the input TDM.
    '''
    term_df = DataFrame(tdm.sum(), columns = ['frequency'])
    term_df['density'] = term_df.frequency / float(term_df.frequency.sum())
    term_df['occurrence'] = tdm.apply(lambda x: np.sum((x > 0))) / float(tdm.shape[0])

    return term_df.sort_index(by = 'occurrence', ascending = False)

spam_term_df = make_term_df(spam_tdm)

print 'Spam Training Set Term Statistics'
print spam_term_df.head()

easyham_tdm = tdm_df(all_easyham, stopwords = rsw, sparse_df = True)

easyham_term_df = make_term_df(easyham_tdm)

print 'Ham Training Set Term Statistics'
print easyham_term_df.head(6)

def classify_email(msg, training_df, prior = 0.5, c = 1e-6):
    '''
    A conditional probability calculator for a naive Bayes e-mail
    classifier.
    Given an e-mail message and a training dataset, the classifier
    returns the log probability of observing the terms in the message if
    it were of the same class as the e-mails in the training set (spam/ham).

    NB: Log probabilities are used for this function, because the raw probabilities
    will be so small that underflow is a real risk. Calculating probability
    would require multiplying many occurrence probabilities -- p1 * p2 * ... * pN,
    where pi is often ~= 0. For log probability we can compute ln(p1) + ln(p2) +
    ... + ln(pN), where ln(pi) < 0 by a far. This will not affect the ordering
    of probabilities (which is what we care about ultimately), but solves the
    underflow risk. Cf. p. 89 of MLFH to see how small raw probability calculations
    can get, and an apparent underflow in row 4.

    Returns a log probability (float) between -Infty and +Infty.
    '''
    msg_tdm = tdm_df([msg])
    msg_freq = msg_tdm.sum()
    msg_match = list(set(msg_freq.index).intersection(set(training_df.index)))
    if len(msg_match) < 1:
        return math.log(prior) + math.log(c) * len(msg_freq)
    else:
        match_probs = training_df.occurrence[msg_match]
        return (math.log(prior) + np.log(match_probs).sum()
                + math.log(c) * (len(msg_freq) - len(msg_match)))

hardham_spamtest = [classify_email(m, spam_term_df) for m in all_hardham]
hardham_hamtest = [classify_email(m, easyham_term_df) for m in all_hardham]
s_spam = np.array(hardham_spamtest) > np.array(hardham_hamtest)

def spam_classifier(msglist):
    '''
    The naive Bayes classifier.
    Using spam and ham training datasets, use `classify_email()` to
    compute the conditional log probability of each e-mail in a list.
    Assign each e-mail to whichever class's training data returns the
    highest probability.

    Returns a DataFrame with the conditional log probabilities and the
    class.
    '''
    spamprob = [classify_email(m, spam_term_df) for m in msglist]
    hamprob = [classify_email(m, easyham_term_df) for m in msglist]
    classify = np.where(np.array(spamprob) > np.array(hamprob), 'Spam', 'Ham')
    out_df = DataFrame({'pr_spam' : spamprob,
                        'pr_ham'  : hamprob,
                        'classify'   : classify},
                       columns = ['pr_spam', 'pr_ham', 'classify'])
    return out_df

def class_stats(df):
    return df.classify.value_counts() / float(len(df.classify))

hardham_classify = spam_classifier(all_hardham)
print 'Hard Ham Classification Statistics (first set)'
print class_stats(hardham_classify)

print 'Hard Ham (first set) classification data'
print hardham_classify.head()

# Run the classifier on the evaluation e-mails in the ham2/spam2
# directories.
all_easyham2 = get_msgdir(easyham2_path)
all_hardham2 = get_msgdir(hardham2_path)
all_spam2 = get_msgdir(spam2_path)

# The classifier does a great job on easy ham.
easyham2_classify = spam_classifier(all_easyham2)
print 'Easy Ham Classification Statistics'
print class_stats(easyham2_classify)

# But it does a pretty bad job on hardham,
# not surprisingly.
hardham2_classify = spam_classifier(all_hardham2)
print 'Hard Ham Classification Statistics'
print class_stats(hardham2_classify)

# It's also very accurate for spam.
spam2_classify = spam_classifier(all_spam2)
print 'Spam Classification Statistics'
print class_stats(spam2_classify)

# These are are almost identical to results using the authors' R
# script after modifying the classify.email() function to use log
# probabilities.
#
#                NOT SPAM       SPAM
# easyham2.col 0.97928571 0.02071429
# hardham2.col 0.30241935 0.69758065
# spam2.col    0.03006442 0.96993558

