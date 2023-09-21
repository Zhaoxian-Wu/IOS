class A():

    def __init__(self, a1, a2) -> None:
        self.a1 = a1
        self.a2 = a2

        self.test(self.a1, self.a2)

    def test(self, a1, a2):
        print(f'test:{a1, a2}')


class B(A):

    def __init__(self, a1, a2) -> None:
        super().__init__(a1, a2)

    def test(self, a1, a2):
        print(f'test: {a1 + a2}')

A1 = A(2, 3)

B1 = B(2, 3)

