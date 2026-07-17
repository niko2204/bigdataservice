from src.education.personalized_data import make_student_dataset, student_seed


def test_same_student_gets_same_data():
    a = make_student_dataset("20260001")
    b = make_student_dataset("20260001")
    assert a.equals(b)


def test_students_get_different_data():
    assert student_seed("20260001") != student_seed("20260002")
    assert not make_student_dataset("20260001").equals(make_student_dataset("20260002"))


def test_quality_problems_are_present():
    df = make_student_dataset("20260001")
    assert df.isna().sum().sum() > 0
    assert df.duplicated().sum() >= 3

