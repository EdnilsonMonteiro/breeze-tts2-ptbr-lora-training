import pytest

pytest.importorskip("num2words")
from text_norm import normalize as n  # noqa: E402


def test_thousands_no_comma():
    assert n("1.234 casos") == "mil duzentos e trinta e quatro casos"


def test_decimal_and_leading_zero():
    assert n("3,14") == "três vírgula catorze"
    assert n("0,05") == "zero vírgula zero cinco"
    assert n("007") == "zero zero sete"


def test_currency_percent():
    assert n("R$ 1.234,56") == "mil duzentos e trinta e quatro reais e cinquenta e seis centavos"
    assert n("R$ 5 mil") == "cinco mil reais"
    assert n("25%") == "vinte e cinco por cento"


def test_time_and_date():
    assert n("às 23h30") == "às vinte e três horas e trinta minutos"
    assert n("1h") == "uma hora"
    assert n("2h") == "duas horas"
    assert n("14:05") == "catorze horas e cinco minutos"
    assert n("01/05/2020") == "primeiro de maio de dois mil e vinte"


def test_ordinals_and_roman():
    assert n("1º lugar") == "primeiro lugar"
    assert n("2ª vez") == "segunda vez"
    assert n("século XX") == "século vinte"
    assert n("Pedro II") == "Pedro segundo"
    assert n("João XXIII") == "João vinte e três"


def test_digit_by_digit():
    assert n("123.456.789-00") == "um dois três quatro cinco seis sete oito nove zero zero"
    assert n("12345678901").split()[:3] == ["um", "dois", "três"]


def test_acronyms():
    assert n("CPF") == "cê pê efe"
    assert n("ATENÇÃO") == "atenção"            # palavra em caixa alta NAO e sigla
    assert n("NÃO FAÇA ISSO") == "não faça isso"
    assert n("PT") == "pê tê"                   # sem vogal -> soletra
    assert n("os CDs") == "os cê dês"


def test_urls_untouched_and_ranges():
    assert n("www.site.com 2 x@y.com") == "www.site.com dois x@y.com"
    assert "menos cinco" in n("-5 graus")
    assert n("Café & pão") == "Café e pão"


def test_idempotent_on_plain_text():
    s = "Ele foi para casa e voltou mais tarde."
    assert n(s) == s
    assert n(n("Foram 1.234 casos às 23h")) == n("Foram 1.234 casos às 23h")
