for typ in [ "dev_1"]:
    infile = typ + ".txt" + ".old"
    blfile = typ + ".txt" + ".block"
    new_text = ''
    with open(infile) as inf, open(blfile) as blf:
        lines = inf.readlines()
        blocks_all = blf.readlines()
        for i, line in enumerate(lines):
            if len(line.split())>1:
                # print(i)

                blocks = [int(item) for item in blocks_all[i].split() ]
                lst = line.rstrip().split(' ', 1)
                label = lst[0]
                tokens = lst[1].split(' ')
                assert len(tokens) == len(blocks)
                len_sst = sum(blocks)
                new_tokens = tokens + ['[SEP]']
                new_text += label + '\t' + ' '.join(new_tokens) + '\n'

    with open(typ + ".txt", 'w') as outf:
        outf.write(new_text)
