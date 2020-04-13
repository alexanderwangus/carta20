'''
Define a req as taking NUM out of the reqs specified in list REQS.
A req is a list of courses or reqs.
'''
class Req():
    def __init__(self, reqs, num=-1):
        self.reqs = reqs
        self.num = num
        self.all = num == -1


'''
Define a subreq (i.e. "track requirement B") as taking NUM out of the courses specified in
the union of list COURSES and REQ.COURSES.
'''
class Subreq():
    def __init__(self, parents, reqs, num=-1):
        self.reqs = [r for p in parents for r in p.reqs]
        self.reqs += reqs
        self.num = num
        self.all = num == -1


'''
Filter out students from dataframe X that do not fulfill given requirement req
'''
def filter_req(X, req):
    if isinstance(req, tuple):
        print(req[0].reqs)
    if req.all:
        for r in req.reqs:
            if isinstance(r, str):
                if r not in rev_vocab:
                    print(r, 'not in rev_vocab')
                    continue
                courseix = rev_vocab[r]
                X = X[X.iloc[:,courseix] > 0]
            elif isinstance(r, Req) or isinstance(r, Subreq):
                stud_idx = filter_req(X, r)
                X = X[X.index.isin(stud_idx)]
            else:
                print(type(r), r.reqs)
                raise TypeError
        return X.index
    else:
        stud_filters = []
        for r in req.reqs:
            if isinstance(r, str):
                if r not in rev_vocab:
                    print(r, 'not in rev_vocab')
                    continue
                courseix = rev_vocab[r]
                stud_filters.append(X.iloc[:,courseix] > 0)
            elif isinstance(r, Req) or isinstance(r, Subreq):
                stud_idx = filter_req(X, r)
                stud_filters.append(X.index.isin(stud_idx))
            else:
                print(type(r), r.reqs)
                raise TypeError
        filter_df = pd.DataFrame(stud_filters).T
        return X.index[filter_df.sum(axis=1) >= req.num]


'''
Wrapper for filter_req
'''
def get_studlist(X, req, filename):
    stud_idx = filter_req(X, req)
    matches = stud_idx.to_series()
    return matches
