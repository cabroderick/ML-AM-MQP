import csv
import json

CSV_PATH = '/home/cabroderick/area_dist.csv'

with open(CSV_PATH) as file:
    areas = csv.reader(file)
    rows = []
    for row in areas:
        if row[0] == 'lack of fusion porosity':
            for elem in row[1:]:
                array = json.loads(elem)
                rows = [row for row in rows if row > 800]
                rows = rows + array

rows.sort()
count = len(rows)
first = rows[int(count/3)]
second = rows[int(2*count/3)]
third = rows[count-1]

print('First threshold: ' + str(first))
print('Second threshold: ' + str(second))
print('Third threshold: ' + str(third))

# first_count, second_count, third_count = 0, 0 ,0
#     print(elem)
#     if elem < first:
#         first_count +=1
#     if elem > first and elem < second:
#         second_count+=1
#     if elem > second and elem < third:
#         third_count+=1

# print(first_count, second_count, third_count)