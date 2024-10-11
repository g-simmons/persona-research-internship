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
    
    def get_synset(self, term):
        for synset in self.all_synsets():
            if synset.name() == term:
                return synset
        return None


class Synset:
    def __init__(self, ontology_class):
        self.ontology_class = ontology_class
    
    def hyponyms(self):
        children = []
        for child in self.ontology_class.subclasses(distance = 1, with_self = False):
            children.append(Synset(child))
        return children
    
    def hypernym_paths(self):
        # for parent in self._get_parents():
        #     print(parent.name())

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
    


# test = Onto("owl/aism.owl")
# print(test.all_synsets())

# counter = 0
# for thing in test.all_synsets():
#     counter += 1
#     print("NAME: " + thing.name())
#     print()

#     for path in thing.hypernym_paths():
#         for syn in path:
#             print(syn.name())
#         print()

#     print("\n\n\n")

#     if counter == 2000:
#         break
# print(counter)

