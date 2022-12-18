from py2neo import Graph, NodeMatcher
from typing import NoReturn


class GraphUse:
    def __init__(self):
        self._graph = Graph("bolt://localhost:7687",
                            auth=("neo4j", "22446688"))  # Initialize DB

    def show_all(self) -> NoReturn:
        print(self._graph.query(
            "MATCH (n)-[rel]->(p)"
            "RETURN n.name as vert_1, type(rel) as relation, p.name as vert_2").to_data_frame())

    def get_image_name(self, name_image: str) -> str:
        if NodeMatcher(self._graph).match("Gesture", name=name_image).exists():
            name_gesture = self._graph.query(f"match (n:Gesture)<-[rel]->(p)"
                                             f"where n.name = \"{name_image}\" "
                                             f"return p.name").evaluate()
            return name_gesture
        else:
            return "Нет такого жеста"