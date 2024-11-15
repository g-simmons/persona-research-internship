#!/usr/bin/env python3

import owlready2
import pronto


# go = pronto.Ontology("owl/aism.owl")


class Onto:
    def __init__(self, owl_dir):
        # go = pronto.Ontology(owl_dir)
        self.onto = pronto.Ontology(owl_dir)

    def all_synsets(self):
        return [Synset(term) for term in self.onto.terms()]
    
    def get_synset(self, term):     # gets the Synset object from the term name
        for synset in self.all_synsets():
            if synset.name() == term:
                return synset
        return None


class Synset:
    def __init__(self, ontology_class):
        self.ontology_class = ontology_class        # pronto term object
    
    def hyponyms(self):
        children = []
        for child in self.ontology_class.subclasses(distance = 1, with_self = False):
            children.append(Synset(child))
        return children
    
    def hypernym_paths(self):
        return list(map(lambda x: reversed(x), Synset._get_paths(self.ontology_class)))
        
    @staticmethod
    def _get_paths(term_obj):
        parents = Synset._get_parents(term_obj)
        if len(parents) == 0:
            return [[Synset(term_obj)]]

        paths = []
        for parent in parents:
            parent_path = Synset._get_paths(parent.ontology_class)
            for path in parent_path:
                paths.append([Synset(term_obj)] + path)
            
        return paths

    @staticmethod
    def _get_parents(term_obj):
        parents = []
        for parent in term_obj.superclasses(distance = 1, with_self = False):
            parents.append(Synset(parent))
        return parents
    
    def lemmas(self):
        synonyms = []
        for synonym in self.ontology_class.synonyms:
            rep_str = synonym.__repr__()
            synonyms.append(Lemma(rep_str.split("'")[1]))

        return synonyms

    def get_depth(self, type = "mean"):
        depths = []
        for path in self.hypernym_paths():
            depths.append(len(list(path)))

        depths = list(map(lambda x: x-1, depths))

        if type == "max":
            return max(depths)
        elif type == "min":
            return min(depths)
        elif type == "mean":
            return sum(depths) / len(depths)
        
    def get_ontology_class(self):
        return list(map(lambda x: x.name(), self.hypernym_paths()[0]))[0]
    
    def name(self):
        if self.ontology_class.name != None:
            new_str = self.ontology_class.name.replace(" ", "_") 
            return new_str
        else:
            return self.ontology_class.name
    
class Lemma:
    def __init__(self, term):
        self.term = term

    def name(self):
        new_str = self.term.replace(" ", "_")
        return new_str
    


if __name__ == "__main__":
    test = Onto("owl/mp.owl")
    synsets = test.all_synsets()


    rand = test.get_synset("Mammalia")
    print(rand.name())
    # for path in rand.hypernym_paths():
    print(list(map(lambda x: x.name(), rand.hypernym_paths()[0])))
    print(rand.get_ontology_class())

