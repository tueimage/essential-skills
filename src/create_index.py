import sys


path = sys.argv[1]
print(path)

with open(path) as f:
    lines = [x for x in f.readlines() if x[0] == '#']

sections = []
for line in lines:
    words = line.split(' ')
    indent = words[0].replace('#', '\t') + '* '
    sections.append(indent + '[' +  ' '.join(words[1:]).strip() + ']' + '(#' + '-'.join(words[1:]).strip().lower() + ')')

minimal_indent = min([len(line.split(' ')[0]) for line in sections]) - 1

sections = [k[minimal_indent:] for k in sections]


print('**Contents**\n')
print('\n'.join(sections))
