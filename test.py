import pandas as pd


def convert_sentence_to_counts(sentence):
    words = sentence.split()

    bed_words = 0
    good_words = 0
    nautral_words = 0

    # read good or bed words from file.
    bedwords = []

    with open('badwords.txt', 'r') as bw:
        bwl = bw.readlines()
        for bword in bwl:
            bedwords.append(bword.strip())

    for word in words:
        if word in bedwords:
            bed_words += 1

    return bed_words


if __name__ == "__main__":
    train_file = ""
    convert_sentence_to_counts("fuck off")
    df = pd.read_csv(train_file)
    df['bed_count'] = df['comments'].apply(convert_sentence_to_counts)

hello world how are you
