class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()

    def number(self, value):
        n = self.v2n.get(value)

        if n is None:
            n = len(self.n2v)
            self.v2n[value] = n
            self.n2v.append(value)

        return n

    def value(self, number):
        return self.n2v[number]

    def max_number(self):
        return len(self.n2v)
