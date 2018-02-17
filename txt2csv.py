import csv

output = []

with open('fm_submissions.txt') as f:
    for line in f.readlines():
        line = line.strip('\n')
        tokens = line.split()
        output.append((tokens[0], tokens[1]))


csv_file = file('fm_submissions.csv', 'wb')
writer = csv.writer(csv_file, delimiter=' ')
writer.writerows(output)
csv_file.close()