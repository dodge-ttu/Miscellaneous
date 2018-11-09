class a(object):

    def get_this(self):
        this_ls = []
        for i in range(10):
            a = i +1
            b = i +2
            c = i +3
            this_ls.append((a,b,c))

        self.something_ls = this_ls


bla = a()


bla.get_this()
