import sys
import calendar

def month_extraction(birthdate):
    date = [int(i) for i in birthdate if i != '/']
    index_month = 10*date[2]+date[3]
    return calendar.month_name[index_month]

class Person:
    def __init__(self, name, surname, birthplace, birthdate):
        self.name = name
        self.surname = surname
        self.birthplace = birthplace
        self.birthdate = birthdate

with open(sys.argv[1], 'r') as f:
    city_births = {}
    month_births = {}
    total_births = 0
    for line in f:
        total_births += 1
        name, surname, birthplace, birthdate = line.split()
        person = Person(name, surname, birthplace, birthdate)

        if person.birthplace not in city_births:
            city_births[person.birthplace] = 0
        city_births[person.birthplace] += 1

        month = month_extraction(person.birthdate)
        if month not in month_births:
            month_births[month] = 0
        month_births[month] += 1

    print("Births per city:")
    for key1, value1 in city_births.items():
        print('  %s: %s' % (key1, value1))

    print("Births per month:")
    for key, value in month_births.items():
        print('  %s: %s' % (key, value))

    print("Average number of births: %.2f" % (total_births/len(city_births)))

