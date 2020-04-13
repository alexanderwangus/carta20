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
