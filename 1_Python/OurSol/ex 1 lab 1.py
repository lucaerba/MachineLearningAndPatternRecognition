import sys

def final_ranking(scores):
    return sum(sorted(scores)[1:-1])

class Competitors:
    def __init__(self, name, surname, nat, scores):
        self.name = name
        self.surname = surname
        self.nat = nat
        self.points = scores
        self.finalscore = final_ranking(self.points)


with open(sys.argv[1], 'r') as f:
    list_best_comp = []
    country_scores = {}
    for line in f:
        name, surname, nat = line.split()[0:3]
        scores = line.split()[3:]
        scores = [float(i) for i in scores]
        comp = Competitors(name, surname, nat, scores)

        list_best_comp.append(comp)
        list_best_comp = sorted(list_best_comp, key = lambda i: i.finalscore)[::-1][0:3]

        if comp.nat not in country_scores:
            country_scores[comp.nat] = 0
        country_scores[comp.nat] += comp.finalscore
        country_scores = {key : value for key, value in sorted(country_scores.items(), key= lambda i: i[1])[::-1]}

    for pos, comp in enumerate(list_best_comp):
        print('%d: %s %s -- Final score: %.1f' % (pos+1, comp.name, comp.surname, comp.finalscore))

    print()
    print('%s -- Total score: %.1f' % (list(country_scores.keys())[0], list(country_scores.values())[0]) )



