import os
import nltk
# nltk.download('averaged_perceptron_tagger')

closed_class_categories = ['CD', 'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP',
                           ' PRP$', 'RP', 'TO', 'UH', 'WDT', 'WP', 'WP$', 'WRB', '', ':', '.', ',']

read_path = 'txt_files/'
write_path = 'tag_files/'

entries = os.listdir(read_path)
ide = 0

for i in entries:
    r_sub_path = read_path + i

    with open(r_sub_path, 'r') as f:
        this_file = f.read()
    f.close()

    this_file = nltk.word_tokenize(this_file)
    tags = nltk.pos_tag(this_file)

    tag_name = write_path + str(ide)
    with open(tag_name, 'w') as f:
        for t in tags:
            if t[1] not in closed_class_categories:
                f.write(t[0] + '\t' + t[1] + '\n')
    f.close()
    ide = ide + 1

