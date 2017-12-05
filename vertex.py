#!/usr/bin/python
# -*- coding: UTF-8 -*-

class Vertex:
    verCount = 0

    def __init__(self, name, type):
        self.name = name
        self.type = type
        Vertex.verCount += 1

    def __hash__(self):
        seed = 131
        result = 0
        key = str(self.type)+"_"+str(self.name)
        for k in key:
            result = result*seed + ord(k)
        return  result