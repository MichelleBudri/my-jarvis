from backend.audio.sentences import SentenceSplitter, clean_for_speech


def feed_all(splitter, tokens):
    out = []
    for t in tokens:
        out += splitter.feed(t)
    if rest := splitter.flush():
        out.append(rest)
    return out


def test_splits_streamed_tokens_into_sentences():
    tokens = ["Boa noite, ", "senhora. A tempe", "ratura é de 18 graus! ", "Deseja algo mais?"]
    assert feed_all(SentenceSplitter(10), tokens) == [
        "Boa noite, senhora.",
        "A temperatura é de 18 graus!",
        "Deseja algo mais?",
    ]


def test_first_sentence_goes_out_early_then_short_ones_merge():
    tokens = ["Boa noite. ", "Sim. ", "Claro, senhora. ", "Já vou."]
    assert feed_all(SentenceSplitter(20), tokens) == [
        "Boa noite.",
        "Sim. Claro, senhora.",
        "Já vou.",
    ]


def test_long_first_sentence_breaks_at_a_comma():
    tokens = [
        "A lua é um satélite natural da Terra, ",
        "que completa uma órbita ",
        "em 27,3 dias. ",
    ]
    tokens += ["Sua gravidade é menor, ", "cerca de um sexto."]
    assert feed_all(SentenceSplitter(20), tokens) == [
        "A lua é um satélite natural da Terra,",
        "que completa uma órbita em 27,3 dias.",
        "Sua gravidade é menor, cerca de um sexto.",
    ]


def test_abbreviations_and_decimals_do_not_split():
    text = "O Dr. Silva disse que o modelo qwen3.5 roda bem. Fim da história."
    assert feed_all(SentenceSplitter(5), [text]) == [
        "O Dr. Silva disse que o modelo qwen3.5 roda bem.",
        "Fim da história.",
    ]


def test_clean_for_speech():
    assert clean_for_speech("**Olá**, senhora! 😀 Veja `isto`.") == "Olá, senhora! Veja isto."


def test_clean_for_speech_drops_list_markers():
    text = "Três notícias:\n1. Robôs dobram roupas.\n2) OpenAI lança modelo.\n- Fim."
    assert clean_for_speech(text) == "Três notícias: Robôs dobram roupas. OpenAI lança modelo. Fim."
    assert clean_for_speech("2. Segunda notícia.") == "Segunda notícia."
    assert clean_for_speech("Em 2026. Ou 3,5 graus.") == "Em 2026. Ou 3,5 graus."


def test_clean_for_speech_spells_out_units():
    assert clean_for_speech("Bateria em 89%, 14°C e vento de 10 km/h.") == (
        "Bateria em 89 por cento, 14 graus e vento de 10 quilômetros por hora."
    )
    assert clean_for_speech("It is 14 °C, 30 %.", "en") == "It is 14 degrees, 30 per cent."
    assert clean_for_speech("100% certo") == "100 por cento certo"
    assert clean_for_speech("Chuva de 2,9 mm.") == "Chuva de 2,9 milímetros."
    assert clean_for_speech("Sem número: % e °") == "Sem número: % e"  # lone ° is dropped
