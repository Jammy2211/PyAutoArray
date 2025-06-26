import logging

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Preloads:

    def __init__(self, mapper_index_list = None):

        self.mapper_index_list = mapper_index_list