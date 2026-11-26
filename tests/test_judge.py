from judge import kappa

def test_kappa():
    # observed agreement 0.75, chance agreement 0.5 -> 0.5
    assert kappa([True, True, False, False], [True, False, False, False]) == 0.5
    assert kappa([True, False, True], [True, False, True]) == 1.0
    assert kappa([True, True], [True, True]) == 1.0
