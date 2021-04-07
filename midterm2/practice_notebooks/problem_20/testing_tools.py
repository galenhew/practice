#!/usr/bin/env python3

import json
import pickle

from hashlib import sha256

# === Tibble comparisons ===

def make_hash(doc):
    if not isinstance(doc, str):
        doc = str(doc)
    return sha256(doc.encode()).hexdigest()

def check_hash(doc, key):
    return make_hash(doc) == key

def canonicalize_tibble(X, remove_index=True):
    var_names = sorted(X.columns)
    Y = X[var_names].copy()
    Y.sort_values(by=var_names, inplace=True)
    Y.reset_index(drop=remove_index, inplace=True)
    return Y

def assert_tibbles_left_matches_right(A, B,
                                      exact=False, hash_A=False, verbose=False):
    from pandas.testing import assert_frame_equal
    A_canonical = canonicalize_tibble(A if not hash_A else A.applymap(make_hash),
                                      not verbose)
    B_canonical = canonicalize_tibble(B, not verbose)
    assert_frame_equal(A_canonical, B_canonical, check_exact=exact)

def assert_tibbles_are_equivalent(A, B, **kwargs):
    assert_tibbles_left_matches_right(A, B, **kwargs)

# === File I/O ===

def data_fn(basename, dirname="./resource/asnlib/publicdata/"):
    return f"{dirname}{basename}"

def file_exists(fn, data_fn=data_fn):
    from os.path import isfile
    return isfile(data_fn(fn))

def load_db(fn, data_fn=data_fn, verbose=True):
    from sqlite3 import connect
    indb = data_fn(fn)
    conn_str = f'file:{indb}?mode=ro'
    if verbose:
        print(f"Opening database, '{indb}` ...\n\t[connection string: '{conn_str}']")
    return connect(conn_str, uri=True)

def get_db_schema(conn):
    query = "SELECT sql FROM sqlite_master WHERE type='table';"
    cursor = conn.cursor()
    return cursor.execute(query).fetchall()

def load_json(fn, data_fn=data_fn):
    infile = data_fn(fn)
    with open(infile, "rt") as fp:
        data = json.load(fp)
    print(f"'{infile}': {len(data)}")
    return data

def inspect_json(v):
    from json import dumps
    print(dumps(v, indent=4))

def save_json(data, fn, data_fn=data_fn):
    outfile = data_fn(fn)
    with open(outfile, "wt") as fp:
        json.dump(data, fp, indent=4)
    print(f"'{outfile}': {len(data)}")

def save_df(df, fn, data_fn=data_fn):
    from pandas import DataFrame
    assert isinstance(df, DataFrame)
    outfile = data_fn(fn)
    df.to_csv(outfile, index=False)

def load_pickle(fn, data_fn=data_fn):
    infile = data_fn(fn)
    with open(infile, 'rb') as fp:
        p = pickle.load(fp)
    return p

def save_pickle(pyobj, fn, data_fn=data_fn):
    outfile = data_fn(fn)
    with open(outfile, 'wb') as fp:
        pickle.dump(pyobj, fp)

# === Random word generation ===

def load_freq_table(infile='english_letter_pair_frequencies.txt'):
    from collections import defaultdict
    freq_table = defaultdict(dict)
    with open(data_fn(infile), 'rt') as fp:
        header = fp.readline()[:-1].split(',') # read line and strip newline
        for line in fp.readlines():
            line = line[:-1].split(',') # remove newline
            x = line[0]
            for y, p_str in zip(header[1:], line[1:]):
                freq_table[x][y] = float(p_str)
    return freq_table

freq_2nd = load_freq_table()
freq_1st = freq_2nd[' ']

def gen_word(max_len=16, freq_1st=freq_1st, freq_2nd=freq_2nd):
    from random import randint, choices
    def select(p, spaces=True):
        from random import choices
        letters = list(p.keys())
        weights = list(p.values())
        if not spaces:
            letters = letters[:-1]
            weights = weights[:-1]
        return choices(letters, weights=weights)[0]
    s = select(freq_1st, spaces=False)
    while not s[-1].isspace():
        s += select(freq_2nd[s[-1]], spaces=(len(s) >= 2) or (s in ['a', 'i']))
    return s.strip()

# === Notebook-specific testing ===

def new_unique(S, gen):
    x = gen()
    while x in S:
        x = gen()
    return x

def gen_enum_entities(ids, gen):
    entities = {}
    for i in ids:
        entities[new_unique(entities, gen)] = i
    return sorted([(i, x) for x, i in entities.items()], key=lambda x: x[0])

def gen_states(num_states):
    from random import sample, choice
    def gen():
        return ''.join([choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2)])
    ids = sample(range(100), num_states)
    return gen_enum_entities(ids, gen)

def capitalize(s):
    if len(s) > 0:
        s = s[0].upper() + s[1:]
    return s

def gen_counties(num_counties, st_id):
    from random import sample
    ids = sample(range(st_id*1000, (st_id+1)*1000), num_counties)
    def gen():
        return capitalize(gen_word()) + ' County'
    return gen_enum_entities(ids, gen)

def gen_country(max_states=3, max_counties=3, cmap=None):
    from random import randint
    states = gen_states(randint(1, max_states))
    counties = []
    for i, st in states:
        counties_st = gen_counties(randint(1, max_counties), i)
        if cmap is not None:
            cmap[st] = counties_st
        counties += counties_st
    return states, counties

def mt2_ex0__gen_db(conn=None):
    if conn is None:
        from sqlite3 import connect
        conn = connect(':memory:')
    cmap = {}
    states, counties = gen_country(cmap=cmap)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS States')
    cursor.execute('CREATE TABLE States (id INTEGER, name TEXT)')
    cursor.executemany('INSERT INTO States VALUES (?, ?)', states)
    cursor.execute('CREATE TABLE Counties (id INTEGER, name TEXT)')
    cursor.executemany('INSERT INTO Counties VALUES (?, ?)', counties)
    conn.commit()
    return conn, states, counties, cmap

def mt2_ex0__check(count_counties):
    from pandas import read_sql_query
    conn, _, _, cmap = mt2_ex0__gen_db()
    soln = {s: len(c) for s, c in cmap.items()}
    try:
        your_soln = count_counties(conn)
        assert set(soln.keys()) == set(your_soln.keys()), \
               f"The keys (states) of your solution are incorrect."
        for st in soln:
            assert soln[st] == your_soln[st], \
                   f"For state '{st}', your count is {your_soln[st]} instead of {soln[st]}."
    except:
        print("=== Test case ===")
        print("* Input `States` table for testing:")
        query = 'SELECT * FROM States'
        display(read_sql_query(query, conn))
        print("* Input `Counties` table for testing:")
        query = 'SELECT * FROM Counties'
        display(read_sql_query(query, conn))
        print("* Expected output:")
        inspect_json(soln)
        if 'your_soln' in locals():
            print("\n* Your output:")
            inspect_json(your_soln)
        raise
    finally:
        conn.close()

def gen_partition(x, n_max):
    """
    Partitions the integer `x` into the sum `x = s_0 + s_1 + ... + s_k`,
    where `k = min(x, n_max)` and each `s_i >= 1` is selected randomly.
    """
    from random import randint
    k = min(x, n_max)
    x_avail = x
    s = []
    for i in range(k-1):
        s_i = 1 + randint(0, x_avail-(k-i))
        x_avail -= s_i
        s.append(s_i)
    assert x_avail >= 1, f"x_avail == {x_avail} < 1"
    s.append(x_avail)
    assert sum(s) == x, f"sum(s) == sum({s}) == {sum(s)} < {x}"
    return s

def gen_flows(counties, year, edges=None, incomes=None, returns=None):
    from collections import defaultdict
    from random import randint, sample
    
    def insert_edge(edges, i, j, f):
        agi = randint(1, 100_000) # income
        edges[i].append((j, year, f, agi))
        if i == j:
            incomes[i] += agi
            returns[i] += f
        else:
            incomes[i] += agi / 2
            incomes[j] += agi / 2
            returns[i] += f / 2
            returns[j] += f / 2
            
    if edges is None:
        edges = defaultdict(list)
    if incomes is None:
        incomes = defaultdict(float)
    if returns is None:
        returns = defaultdict(float)
    for c in counties:
        pop_c = randint(1, 1_000_000) # total population of `c`
        if len(counties) > 1:
            self_flow = randint(1, pop_c) # number who *do not* move
        else:
            self_flow = pop_c
        insert_edge(edges, c, c, self_flow)
        total_out_flow = pop_c - self_flow # number who *do* move
        if total_out_flow > 0:
            # determine number and size of out_flows
            out_flows = gen_partition(total_out_flow, len(counties)-1)
            num_edges = len(out_flows)
            other_counties = list(set(counties) - {c})
            targets = sample(other_counties, num_edges)
            for t, f in zip(targets, out_flows):
                insert_edge(edges, c, t, f)
    return edges, incomes, returns

def mt2_ex1__gen_db(conn=None, counties=None):
    from collections import defaultdict
    from random import randint
    if counties is None:
        if conn is None:
            conn, _, counties, _ = mt2_ex0__gen_db()
        else:
            _, _, counties, _ = mt2_ex0__gen_db(conn)
    elif conn is None:
        from sqlite3 import connect
        conn = connect(':memory:')
    y0 = randint(2011, 2017)
    flows = defaultdict(list)
    incomes = defaultdict(float)
    returns = defaultdict(float)
    for y in range(y0, y0+2):
        gen_flows(counties, y, flows, incomes, returns)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS Flows')
    cursor.execute('CREATE TABLE Flows (source INTEGER, dest INTEGER, year INTEGER, num_returns INTEGER, income_thousands INTEGER)')
    for (i, _), Ei in flows.items():
        for (j, _), y, nij, agij in Ei:
            cursor.execute('INSERT INTO Flows VALUES (?, ?, ?, ?, ?)',
                           (i, j, y, nij, agij))
    conn.commit()
    return conn, flows, incomes, returns

def iter_flows(flows):
    for i, Ei in flows.items():
        for j, y, nij, agi in Ei:
            yield i, j, y, nij, agi

def gen_totals(flows):
    from collections import defaultdict
    totals = defaultdict(int)
    for (i, _), (j, _), _, nij, _ in iter_flows(flows):
        totals[i] += nij
    return totals

def mt2_ex1__check(sum_outflows):
    from pandas import DataFrame, read_sql_query
    sources = []
    total = []
    conn, flows, _, _ = mt2_ex1__gen_db()
    totals = gen_totals(flows)
    soln = DataFrame({'source': list(totals.keys()),
                      'total_returns': list(totals.values())})
    try:
        your_soln = sum_outflows(conn)
        assert isinstance(your_soln, DataFrame), \
               f'Your function should return a `DataFrame`, but instead returned an object of type `{type(your_soln)}`.'
        assert_tibbles_are_equivalent(soln, your_soln, exact=True)
    except:
        print("=== Test case ===")
        print("* Input `Flows` table for testing:")
        query = 'SELECT * FROM Flows'
        display(read_sql_query(query, conn))
        print("* Expected output:")
        display(soln)
        if 'your_soln' in locals():
            print("\n* Your output:")
            display(your_soln)
        raise
    finally:
        conn.close()

def gen_probs(flows):
    from collections import defaultdict
    totals = gen_totals(flows)
    edges = defaultdict(lambda: defaultdict(int))
    for (i, _), (j, _), _, nij, _ in iter_flows(flows):
        edges[i][j] += nij
    probs = defaultdict(lambda: defaultdict(float))
    for i in edges:
        for j in edges[i]:
            probs[i][j] = edges[i][j] / totals[i]
    return probs

def mt2_ex2__check(estimate_probs):
    from pandas import DataFrame, read_sql_query
    conn, flows, _, _ = mt2_ex1__gen_db()
    probs = gen_probs(flows)
    I, J, P = [], [], []
    for i, pi in probs.items():
        for j, pij in pi.items():
            I.append(i)
            J.append(j)
            P.append(pij)
    soln = DataFrame({'source': I, 'dest': J, 'prob': P})
    try:
        your_soln = estimate_probs(conn)
        assert isinstance(your_soln, DataFrame), \
               f'Your function should return a `DataFrame`, but instead returned an object of type `{type(your_soln)}`.'
        assert_tibbles_are_equivalent(soln, your_soln, exact=False)
    except:
        print("=== Test case ===")
        print("* Input `Flows` table for testing:")
        query = 'SELECT * FROM Flows'
        display(read_sql_query(query, conn))
        print("* Expected output:")
        display(soln)
        if 'your_soln' in locals():
            print("\n* Your output:")
            display(your_soln)
        raise
    finally:
        conn.close()

def gen_county_map(counties):
    return {i: k for k, (i, _) in enumerate(counties)}

def mt2_ex3__check(map_counties):
    from pandas import read_sql_query
    conn, _, counties, _ = mt2_ex0__gen_db()
    soln = gen_county_map(counties)
    try:
        your_soln = map_counties(conn)
        assert set(soln.keys()) == set(your_soln.keys()), \
               f"The keys (county IDs) of your solution are incorrect."
        for i in soln:
            assert soln[i] == your_soln[i], \
                   f"For county ID '{i}', your count is {your_soln[i]} instead of {soln[i]}."
    except:
        print("=== Test case ===")
        print("* Input `Counties` table for testing:")
        query = 'SELECT * FROM Counties'
        display(read_sql_query(query, conn))
        print("* Expected output:")
        inspect_json(soln)
        if 'your_soln' in locals():
            print("\n* Your output:")
            inspect_json(your_soln)
        raise
    finally:
        conn.close()

def mt2_ex4__check(make_matrix):
    from pandas import DataFrame
    from scipy.sparse import coo_matrix
    from numpy import isclose
    conn, _, counties, _ = mt2_ex0__gen_db()
    county_map = gen_county_map(counties)
    _, flows, _, _ = mt2_ex1__gen_db(conn=conn, counties=counties)
    probs_dict = gen_probs(flows)
    CI, CJ = [], []
    I, J, P = [], [], []
    for i, pi in probs_dict.items():
        for j, pij in pi.items():
            I.append(county_map[i])
            CI.append(i)
            J.append(county_map[j])
            CJ.append(j)
            P.append(pij)    
    probs = DataFrame({'source': CI, 'dest': CJ, 'prob': P})
    probs_mat = DataFrame({'true_row': I, 'true_col': J, 'true_val': P})
    n = len(county_map)
    try:
        your_soln = make_matrix(probs, county_map)
        empty_coo = coo_matrix(([], ([], [])), shape=(0, 0))
        assert isinstance(your_soln, type(empty_coo)), \
               f"Your function returned an object of type '{type(your_soln)}' instead of a Scipy COO matrix (type {type(empty_coo)})."
        assert your_soln.shape == (n, n), \
               f"Your matrix is {your_soln.shape[0]}x{your_soln.shape[1]} instead of {n}x{n}."
        your_mat = DataFrame({'your_row': your_soln.row,
                              'your_col': your_soln.col,
                              'your_val': your_soln.data})
        combined = probs_mat.merge(your_mat, how='outer',
                                   left_on=['true_row', 'true_col'],
                                   right_on=['your_row', 'your_col'])
        bad_rows = combined.isna().all(axis=1)
        bad_rows |= ~isclose(combined['true_val'], combined['your_val'])
        assert (~bad_rows).all(), f"Nonzero mismatches detected"
    except:
        print("=== Test case ===")
        print("* Input: Logical-to-physical county ID map")
        inspect_json(county_map)
        print("* Input: Probability transitions")
        display(probs)
        if 'combined' in locals() and 'bad_rows' in locals():
            print("* Output: Nonzero mismatches, as a `DataFrame` for ease-of-reading")
            print("  Any row with a `NaN` indicates a nonzero present (or absent) in our solution that are absent (or present) in yours. If you don't see NaNs, the nonzero values might differ by more than what we expect from floating-point roundoff error.")
            display(combined[bad_rows])
        else:
            print("* Expected nonzeros, as a `DataFrame` for ease-of-reading (could not obtain output from your code, so displaying the expected solution only)")
            display(probs_mat)
        raise

def gen_peeps():
    from random import sample, randint
    states, counties = gen_country()
    num_counties = len(counties)
    total_peeps = randint(max(num_counties, 100_000), 100_000_000)
    birth_rate = randint(1, 1_000)/10_000
    death_rate = randint(1, 1_000)/10_000
    total_births = int(total_peeps * birth_rate)
    total_deaths = int(total_peeps * death_rate)
    births = gen_partition(total_births, num_counties)
    deaths = gen_partition(total_deaths, num_counties)
    peeps_min = [max(b-d, 0) for b, d in zip(births, deaths)]
    peeps_avail = total_peeps - sum(peeps_min)
    peeps_add = gen_partition(peeps_avail, num_counties)
    peeps = [p0+dp for p0, dp in zip(peeps_min, peeps_add)]
    return states, counties, \
           total_peeps, birth_rate, death_rate, \
           peeps, births, deaths

def gen_peeps_df(state_ids, county_ids, peeps, births, deaths):
    from pandas import DataFrame
    return DataFrame({'STATE': state_ids,
                      'COUNTY': [i%1000 for i in county_ids],
                      'POPESTIMATE2019': peeps,
                      'BIRTHS2019': births,
                      'DEATHS2019': deaths})

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return tuple(factors)

def mt2_ex5__check(normalize_pop):
    from numpy import empty, isclose, arange
    from pandas import DataFrame
    _, counties, total, beta, delta, peeps, births, deaths = gen_peeps()
    county_ids = [i for i, _ in counties]
    state_ids = [i//1000 for i in county_ids]
    peeps_df = gen_peeps_df(state_ids, county_ids, peeps, births, deaths)
    county_map = gen_county_map(counties)
    x0 = empty(shape=len(county_map))
    for i, ni in zip(county_ids, peeps):
        x0[county_map[i]] = ni/total
    try:
        your_x0 = normalize_pop(peeps_df, county_map)
        assert isinstance(your_x0, type(x0)), \
               f"Your function returned an object of type {type(your_x0)} rather than a Numpy array ({type(x0)})."
        assert your_x0.ndim == x0.ndim, \
               f"You returned a {your_x0.ndim}-dimensional array, whereas we expected a {x0.ndim}-dimensional one."
        assert your_x0.shape == x0.shape, \
               f"The shape of your array is {your_x0.shape} rather than {x0.shape}."
        bad_elems = ~isclose(your_x0, x0)
        assert (~bad_elems).all(), \
               f"Your computed values do not match our solution."
    except:
        print("=== Test case ===")
        print("* Input population data frame:")
        display(peeps_df)
        print("* Input county map:")
        inspect_json(county_map)
        print("* Expected output:")
        print(x0)
        if 'your_x0' in locals():
            print("* Your output:")
            print(your_x0)
            if 'bad_elems' in locals():
                print("* The following elements of your output didn't match ours:")
                mismatches = DataFrame({'position': arange(len(x0))[bad_elems],
                                        'our_value': x0[bad_elems],
                                        'your_value': your_x0[bad_elems]})
        raise
    
def mt2_ex6__check(estimate_pop):
    from random import randint
    from math import isclose
    _, counties, total, beta, delta, peeps, births, deaths = gen_peeps()
    county_ids = [i for i, _ in counties]
    state_ids = [i//1000 for i in county_ids]
    peeps_df = gen_peeps_df(state_ids, county_ids, peeps, births, deaths)
    t = randint(1, 100)
    total_t = total * (1 + beta - delta)**t
    try:
        your_total_t = estimate_pop(peeps_df, t)
        assert isinstance(your_total_t, float), \
               f"Your function returned an object of type `{type(your_total_t)}` instead of a `float`."
        rel_err = abs(your_total_t - total_t)/total_t
        assert rel_err <= 1e-2, \
               f"Your value of {your_total_t} is not close enough to our estimate, {total_t} (relative error={rel_err})."
    except:
        print("=== Test case ===")
        print("* Input data frame:")
        display(peeps_df)
        print(f"* Future time value: t={t} years")
        print(f"* Expected output: {total_t}")
        if 'your_total_t' in locals():
            print(f"* Your output: {your_total_t}")
        raise

def mt2_ex7__check(calc_ipr):
    from pandas import DataFrame, read_sql_query
    conn, _, incomes, returns = mt2_ex1__gen_db()
    counties = list(incomes.keys())
    try:
        ipr = DataFrame({'county_id': [i for i, _ in counties], 'ipr': [1e3*incomes[c]/returns[c] for c in counties]})
    except:
        print('c =', c)
        print('counties =', counties)
        print('incomes =', incomes)
        print('returns =', returns)
        raise
    try:
        your_ipr = calc_ipr(conn)
        assert isinstance(your_ipr, DataFrame), \
               f'Your function should return a `DataFrame`, but instead returned an object of type `{type(your_ipr)}`.'
        assert_tibbles_are_equivalent(ipr, your_ipr, exact=False)
    except:
        print("=== Test case ===")
        print("* Input `Flows` table for testing:")
        query = 'SELECT * FROM Flows'
        display(read_sql_query(query, conn))
        print("* Expected output:")
        display(ipr)
        if 'your_ipr' in locals():
            print("\n* Your output:")
            display(your_ipr)
        raise
    finally:
        conn.close()
    
# eof
