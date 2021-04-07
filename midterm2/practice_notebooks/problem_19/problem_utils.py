#!/usr/bin/env python3

import os
from hashlib import sha256

def get_path(basename):
    from os.path import isdir
    return f'./resource/asnlib/publicdata/{basename}' #if isdir('.voc') else basename

def display_image(image_filename):
    from IPython.display import Image
    return Image(image_filename, width=360)
    
def open_file(basename, *args):
    return open(get_path(basename), *args)

def make_hash(doc):
    return sha256(doc.encode()).hexdigest()

def check_hash(doc, key):
    return make_hash(doc) == key

def where_strings_differ(a, b):
    first_difference = None
    for k, (a_k, b_k) in enumerate(zip(a, b)):
        if a_k != b_k:
            first_difference = k
            break
    if first_difference is None: # strings match up to min(len(a), len(b)), so check for extra chars
        if len(a) < len(b):
            print(f'Strings are the same except the second one has these extra letters: "{b[len(a):]}"\n')
            first_difference = len(a)
        elif len(b) < len(a):
            print(f'Strings are the same except the first one has these extra letters: "{a[len(b):]}"\n')
            first_difference = len(b)
    if first_difference is None:
        print("==> Strings appear to be identical.")
    else:
        print(f"==> Strings differ starting at position {first_difference}:")
        start_snip = max(0, first_difference-5)
        snip_prefix = "..." if start_snip > 0 else ''
        snip_a = snip_prefix + a[start_snip:first_difference+1]
        snip_b = snip_prefix + b[start_snip:first_difference+1]
        print(f"    {snip_a} <-- position {first_difference}")
        print(f"vs.")
        print(f"    {snip_b} <--")
    return first_difference

def canonicalize_tibble(X, remove_index):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=remove_index, inplace=True)
    return Y

def assert_tibbles_are_equivalent(A, B):
    from pandas.testing import assert_frame_equal
    A_c = canonicalize_tibble(A, True)
    B_c = canonicalize_tibble(B, True)
    if len(A_c) == 0 and len(B_c) == 0: return
    assert_frame_equal(A_c, B_c, check_index_type=False, check_less_precise=True)

def pandas_df_to_markdown_table(df, index=False):
    """Adapted from:
    https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    """
    from IPython.display import Markdown, display
    from pandas import DataFrame, concat
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = DataFrame([fmt], columns=df.columns)
    df_formatted = concat([df_fmt, df])
    csv_rep = df_formatted.to_csv(sep="|", index=index)
    if index:
        csv_rep = '(index)' + csv_rep # + csv_rep.replace('\n0|', '\n---|')
    display(Markdown(csv_rep))
    return csv_rep

def hidden_cell_template_msg():
    return "This test cell will be replaced with one or more hidden tests.\n" \
           "You will only know the result after submitting to the autograder.\n" \
           "If the autograder times out, then either your solution is highly\n" \
           "inefficient or contains a bug (e.g., an infinite loop). To see\n" \
           "the result of the autograder run, inspect the grading report.\n"

# eof
