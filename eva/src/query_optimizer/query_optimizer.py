"""
This file composes the functions that are needed to perform query optimization.
Currently, given a query, it does logical changes to forms that are
sufficient conditions.
Using statistics from Filters module, it outputs the optimal plan (converted
query with models needed to be used).

To see the query optimizer performance in action, simply run

python query_optimizer/query_optimizer.py

@Jaeho Bang

"""
import os
import socket
# The query optimizer decide how to label the data points
# Load the series of queries from a txt file?
import sys
import threading
from itertools import product

import numpy as np

from src import constants

eva_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(eva_dir)


class QueryOptimizer:
    """
    TODO: If you have a classifier for =, you can make a classifier for !=
    TODO: Deal with parenthesis
    """

    def __init__(self, ip_str="127.0.0.1"):
        self.ip_str = ip_str
        # self.startSocket()
        self.operators = ["!=", ">=", "<=", "=", "<", ">"]
        self.separators = ["||", "&&"]

    def startSocket(self):
        thread = threading.Thread(target=self.inputQueriesFromSocket)
        thread.daemon = True
        thread.start()
        while True:
            input = eval(input(
                'Type in your query in the form of __label__ > __number__\n'))

            self.parseInput(input)

    def parseInput(self, input):
        """
        TODO: Need to provide query formats that can be used
        :param input: string to be parsed
        :return: something that the Load() class can understand
        """
        pass

    def inputQueriesFromTxt(self, input_path):
        """
        TODO: Read the file line by line, use self.parseInput to give back
        commands
        :param input_path: full directory + file name
        :return: method of training the pps
        """
        pass

    def inputQueriesFromSocket(self):
        sock = socket.socket()
        sock.bind(self.ip_str, 123)
        sock.listen(3)
        print("Waiting on connection")
        conn = sock.accept()
        print("Client connected")
        while True:
            m = conn[0].recv(4096)
            conn[0].send(m[::-1])

        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def _findParenthesis(self, query):

        start = []
        end = []
        query_copy = query
        index = query_copy.find("(")
        while index != -1:
            start.append(index)
            query_copy = query_copy[index + 1:]
            index = query_copy.find("(")

        query_copy = query
        index = query_copy.find(")")
        while index != -1:
            end.append(index)
            query_copy = query_copy[index + 1:]
            index = query_copy.find(")")

        return [start, end]

    def _parseQuery(self, query):
        """
        Each sub query will be a list
        There will be a separator in between
        :param query:
        :return:
        """

        query_parsed = []
        query_subs = query.split(" ")
        query_operators = []
        for query_sub in query_subs:
            if query_sub == "||" or query_sub == "&&":
                query_operators.append(query_sub)
            else:

                if True not in [operator in self.operators for operator in
                                query_sub]:
                    return [], []
                for operator in self.operators:
                    query_sub_list = query_sub.split(operator)
                    if type(query_sub_list) is list and len(
                            query_sub_list) > 1:
                        query_parsed.append(
                            [query_sub_list[0], operator, query_sub_list[1]])
                        break
        # query_parsed ex: [ ["t", "=", "van"], ["s", ">", "60"]]
        # query_operators ex: ["||", "||", "&&"]
        return query_parsed, query_operators

    def _logic_reverse(self, str):
        if str == "=":
            return "!="
        elif str == "!=":
            return "="
        elif str == ">":
            return "<="
        elif str == ">=":
            return "<"
        elif str == "<":
            return ">="
        elif str == "<=":
            return ">"

    def convertL2S(self, parsed_query, query_ops):
        final_str = ""
        index = 0
        for sub_parsed_query in parsed_query:
            if len(parsed_query) >= 2 and index < len(query_ops):
                final_str += ''.join(sub_parsed_query) + " " + query_ops[
                    index] + " "
                index += 1
            else:
                final_str += ''.join(sub_parsed_query)
        return final_str

    def _wrangler(self, query, label_desc):
        """
        import itertools
        iterables = [ [1,2,3,4], [88,99], ['a','b'] ]
        for t in itertools.product(*iterables):
          print t

        Different types of checks are performed
        1. not equals check (f(C) != v)
        2. comparison check (f(C) > v -> f(C) > t, for all t <= v)
        3. Range check (v1 <= f(C) <= v2) - special type of comparison check
        4. No-predicates = when column in finite and discrete, it can still
        benefit
          ex) 1 <=> type = car U type = truck U type = SUV
        :return: transformed query
        """
        # TODO: Need to implement range check

        query_parsed, query_operators = self._parseQuery(query)
        # query_sorted = sorted(query_parsed)

        query_transformed = []
        equivalences = []
        equivalences_op = []

        for query_sub_list in query_parsed:
            subject = query_sub_list[0]
            operator = query_sub_list[1]
            object = query_sub_list[2]

            assert (
                subject in label_desc)  # Label should be in label
            # description dictionary
            l_desc = label_desc[subject]
            if l_desc[0] == constants.DISCRETE:
                equivalence = [self.convertL2S([query_sub_list], [])]
                assert (operator == "=" or operator == "!=")
                alternate_string = ""
                for category in l_desc[1]:
                    if category != object:
                        alternate_string += subject + self._logic_reverse(
                            operator) + category + " && "
                alternate_string = alternate_string[
                    :-len(" && ")]  # must strip the last ' || '
                # query_tmp, _ = self._parseQuery(alternate_string)
                equivalence.append(alternate_string)

            elif l_desc[0] == constants.CONTINUOUS:

                equivalence = [self.convertL2S([query_sub_list], [])]
                assert (operator == "=" or operator == "!=" or operator == "<"
                        or operator == "<=" or operator == ">" or operator ==
                        ">=")
                alternate_string = ""
                if operator == "!=":
                    alternate_string += subject + ">" + object + " && " + \
                        subject + "<" + object
                    query_tmp, _ = self._parseQuery(alternate_string)
                    equivalence.append(query_tmp)
                if operator == "<" or operator == "<=":
                    object_num = eval(object)
                    for number in l_desc[1]:
                        if number > object_num:
                            alternate_string = subject + operator + str(number)
                            # query_tmp, _ = self._parseQuery(alternate_string)
                            equivalence.append(alternate_string)
                if operator == ">" or operator == ">=":
                    object_num = eval(object)
                    for number in l_desc[1]:
                        if number < object_num:
                            alternate_string = subject + operator + str(number)
                            # query_tmp, _ = self._parseQuery(alternate_string)
                            equivalence.append(alternate_string)

            equivalences.append(equivalence)

        possible_queries = product(*equivalences)
        for q in possible_queries:
            query_transformed.append(q)

        return query_transformed, query_operators

    def _compute_expression(self, query_info, pp_list, pp_stats, k,
                            accuracy_budget):
        """

        def QueryOptimizer(P, {trained PPs}):
          P = wrangler(P)
          {E} = compute_expressions(P,{trained PP},k)        #k is a fixed
          constant which limits number of individual PPs
          in the final expression
          for E in {E}:
          Explore_PP_accuracy_budget(E)  # Paper says dynamic program
          Explore_PP_Orderings(E)        #if k is small, any number of orders
          can be explored
          Compute_cost_vs_red_rate(E)   #arithmetic over individual c,
          a and r[a] numbers
          return E_with_max_c/r


        1. p^(P/p) -> PPp
        2. PPp^q -> PPp ^ PPq
        3. PPpvq -> PPp v PPq
        4. p^(P/p) -> ~PP~q
        -> we don't need to apply these rules, we simply need to see for each
        sub query which PP gives us the best rate
        :param query_info: [possible query forms for a given query, operators
        that go in between]
        :param pp_list: list of pp names that are currently available
        :param pp_stats: list of pp models associated with each pp name with
        R,C,A values saved
        :param k: number of pps we can use at maximum
        :return: the list of pps to use that maximizes reduction rate (ATM)
        """
        evaluations = []
        evaluation_models = []
        evaluations_stats = []
        query_transformed, query_operators = query_info
        # query_transformed = [[["t", "!=", "car"], ["t", "=", "van"]], ... ]
        for possible_query in query_transformed:
            evaluation = []
            evaluation_stats = []
            k_count = 0
            op_index = 0
            for query_sub in possible_query:  # Even inside query_sub it can
                # be divided into query_sub_sub
                if k_count > k:  # TODO: If you exceed a certain number,
                    # you just ignore the expression
                    evaluation = []
                    evaluation_stats = []
                    continue
                query_sub_list, query_sub_operators = self._parseQuery(
                    query_sub)
                evaluation_tmp = []
                evaluation_models_tmp = []
                evaluation_stats_tmp = []
                for i in range(len(query_sub_list)):
                    query_sub_str = ''.join(query_sub_list[i])
                    if query_sub_str in pp_list:
                        # Find the best model for the pp

                        data = self._find_model(query_sub_str, pp_stats,
                                                accuracy_budget)
                        if data is None:
                            continue
                        else:
                            model, reduction_rate = data
                            evaluation_tmp.append(query_sub_str)
                            evaluation_models_tmp.append(
                                model)  # TODO: We need to make sure this is
                            # the model_name
                            evaluation_stats_tmp.append(reduction_rate)
                            k_count += 1

                reduc_rate = 0
                if len(evaluation_stats_tmp) != 0:
                    reduc_rate = self._update_stats(evaluation_stats_tmp,
                                                    query_sub_operators)

                evaluation.append(query_sub)
                evaluation_models.append(evaluation_models_tmp)
                evaluation_stats.append(reduc_rate)
            op_index += 1

            evaluations.append(self.convertL2S(evaluation, query_operators))
            evaluations_stats.append(
                self._update_stats(evaluation_stats, query_operators))

        max_index = np.argmax(np.array(evaluations_stats), axis=0)
        best_query = evaluations[
            max_index]  # this will be something like "t!=bus && t!=truck &&
        # t!=car"
        best_models = evaluation_models[max_index]
        best_reduction_rate = evaluations_stats[max_index]

        pp_names, op_names = self._convertQuery2PPOps(best_query)
        return [list(zip(pp_names, best_models)), op_names,
                best_reduction_rate]

    def _convertQuery2PPOps(self, query):
        """

        :param query: str (t!=car && t!=truck)
        :return:
        """
        query_split = query.split(" ")
        pp_names = []
        op_names = []
        for i in range(len(query_split)):
            if i % 2 == 0:
                pp_names.append(query_split[i])
            else:
                if query_split[i] == "&&":
                    op_names.append(np.logical_and)
                else:
                    op_names.append(np.logical_or)

        return pp_names, op_names

    # Make this function take in the list of reduction rates and the operator
    # lists
    def _update_stats(self, evaluation_stats, query_operators):
        if len(evaluation_stats) == 0:
            return 0
        final_red = evaluation_stats[0]
        assert (len(evaluation_stats) == len(query_operators) + 1)

        for i in range(1, len(evaluation_stats)):
            if query_operators[i - 1] == "&&":
                final_red = final_red + evaluation_stats[i] - final_red * \
                    evaluation_stats[i]
            elif query_operators[i - 1] == "||":
                final_red = final_red * evaluation_stats[i]

        return final_red

    def _compute_cost_red_rate(self, C, R):
        assert (
            R >= 0 and R <= 1)  # R is reduction rate and should be
        # between 0 and 1
        if R == 0:
            R = 0.000001
        return float(C) / R

    def _find_model(self, pp_name, pp_stats, accuracy_budget):
        possible_models = pp_stats[pp_name]
        best = []  # [best_model_name, best_model_cost /
        # best_model_reduction_rate]
        for possible_model in possible_models:
            if possible_models[possible_model]["A"] < accuracy_budget:
                continue
            if best == []:
                best = [possible_model, self._compute_cost_red_rate(
                    possible_models[possible_model]["C"],
                    possible_models[possible_model]["R"]),
                    possible_models[possible_model]["R"]]
            else:
                alternative_best_cost = self._compute_cost_red_rate(
                    possible_models[possible_model]["C"],
                    possible_models[possible_model]["R"])
                if alternative_best_cost < best[1]:
                    best = [possible_model, alternative_best_cost,
                            possible_models[possible_model]["R"]]

        if best == []:
            return None
        else:
            return best[0], best[2]

    def run(self, query, pp_list, pp_stats, label_desc, k=3,
            accuracy_budget=0.9):
        """

        :param query: query of interest ex) TRAF-20
        :param pp_list: list of pp_descriptions - queries that are available
        :param pp_stats: this will be dictionary where keys are "pca/ddn",
                         it will have statistics saved which are R (
                         reduction_rate), C (cost_to_train), A (accuracy)
        :param k: number of different PPs that are in any expression E
        :return: selected PPs to use for reduction
        """
        query_transformed, query_operators = self._wrangler(query, label_desc)
        # query_transformed is a comprehensive list of transformed queries
        return self._compute_expression([query_transformed, query_operators],
                                        pp_list, pp_stats, k, accuracy_budget)


if __name__ == "__main__":

    query_list = ["t=suv", "s>60",
                  "c=white", "c!=white", "o=pt211", "c=white && t=suv",
                  "s>60 && s<65", "t=sedan || t=truck", "i=pt335 && o=pt211",
                  "t=suv && c!=white", "c=white && t!=suv && t!=van",
                  "t=van && s>60 && s<65", "c!=white && (t=sedan || t=truck)",
                  "i=pt335 && o!=pt211 && o!=pt208",
                  "t=van && i=pt335 && o=pt211",
                  "t!=sedan && c!=black && c!=silver && t!=truck",
                  "t=van && s>60 && s<65 && o=pt211",
                  "t!=suv && t!=van && c!=red && t!=white",
                  "(i=pt335 || i=pt342) && o!=pt211 && o!=pt208",
                  "i=pt335 && o=pt211 && t=van && c=red"]

    # TODO: Support for parenthesis queries
    query_list_mod = ["t=suv", "s>60",
                      "c=white", "c!=white", "o=pt211", "c=white && t=suv",
                      "s>60 && s<65", "t=sedan || t=truck",
                      "i=pt335 && o=pt211",
                      "t=suv && c!=white", "c=white && t!=suv && t!=van",
                      "t=van && s>60 && s<65",
                      "t=sedan || t=truck && c!=white",
                      "i=pt335 && o!=pt211 && o!=pt208",
                      "t=van && i=pt335 && o=pt211",
                      "t!=sedan && c!=black && c!=silver && t!=truck",
                      "t=van && s>60 && s<65 && o=pt211",
                      "t!=suv && t!=van && c!=red && t!=white",
                      "i=pt335 || i=pt342 && o!=pt211 && o!=pt208",
                      "i=pt335 && o=pt211 && t=van && c=red"]

    query_list_test = ["c=white && t!=suv && t!=van"]

    synthetic_pp_list = ["t=suv", "t=van", "t=sedan", "t=truck",
                         "c=red", "c=white", "c=black", "c=silver",
                         "s>40", "s>50", "s>60", "s<65", "s<70",
                         "i=pt335", "i=pt211", "i=pt342", "i=pt208",
                         "o=pt335", "o=pt211", "o=pt342", "o=pt208"]

    query_list_short = ["t=van && s>60 && o=pt211"]

    synthetic_pp_list_short = ["t=van", "s>60", "o=pt211"]

    # TODO: Might need to change this to a R vs A curve instead of static
    #  numbers
    # TODO: When selecting appropriate PPs, we only select based on reduction
    #  rate
    synthetic_pp_stats_short = {
        "t=van": {"none/dnn": {"R": 0.1, "C": 0.1, "A": 0.9},
                  "pca/dnn": {"R": 0.2, "C": 0.15, "A": 0.92},
                  "none/kde": {"R": 0.15, "C": 0.05, "A": 0.95}},

        "s>60": {"none/dnn": {"R": 0.12, "C": 0.21, "A": 0.87},
                 "none/kde": {"R": 0.15, "C": 0.06, "A": 0.96}},

        "o=pt211": {"none/dnn": {"R": 0.13, "C": 0.32, "A": 0.99},
                    "none/kde": {"R": 0.14, "C": 0.12, "A": 0.93}}}

    synthetic_pp_stats = {"t=van": {"none/dnn": {"R": 0.1, "C": 0.1, "A": 0.9},
                                    "pca/dnn": {"R": 0.2, "C": 0.15,
                                                "A": 0.92},
                                    "none/kde": {"R": 0.15, "C": 0.05,
                                                 "A": 0.95}},
                          "t=suv": {
                              "none/svm": {"R": 0.13, "C": 0.01, "A": 0.95}},
                          "t=sedan": {
                              "none/svm": {"R": 0.21, "C": 0.01, "A": 0.94}},
                          "t=truck": {
                              "none/svm": {"R": 0.05, "C": 0.01, "A": 0.99}},

                          "c=red": {
                              "none/svm": {"R": 0.131, "C": 0.011,
                                           "A": 0.951}},
                          "c=white": {
                              "none/svm": {"R": 0.212, "C": 0.012,
                                           "A": 0.942}},
                          "c=black": {
                              "none/svm": {"R": 0.133, "C": 0.013,
                                           "A": 0.953}},
                          "c=silver": {
                              "none/svm": {"R": 0.214, "C": 0.014,
                                           "A": 0.944}},

                          "s>40": {
                              "none/svm": {"R": 0.08, "C": 0.20, "A": 0.8}},
                          "s>50": {
                              "none/svm": {"R": 0.10, "C": 0.20, "A": 0.82}},

                          "s>60": {
                              "none/dnn": {"R": 0.12, "C": 0.21, "A": 0.87},
                              "none/kde": {"R": 0.15, "C": 0.06, "A": 0.96}},

                          "s<65": {
                              "none/svm": {"R": 0.05, "C": 0.20, "A": 0.8}},
                          "s<70": {
                              "none/svm": {"R": 0.02, "C": 0.20, "A": 0.9}},

                          "o=pt211": {
                              "none/dnn": {"R": 0.135, "C": 0.324, "A": 0.993},
                              "none/kde": {"R": 0.143, "C": 0.123,
                                           "A": 0.932}},

                          "o=pt335": {
                              "none/dnn": {"R": 0.134, "C": 0.324, "A": 0.994},
                              "none/kde": {"R": 0.144, "C": 0.124,
                                           "A": 0.934}},

                          "o=pt342": {
                              "none/dnn": {"R": 0.135, "C": 0.325, "A": 0.995},
                              "none/kde": {"R": 0.145, "C": 0.125,
                                           "A": 0.935}},

                          "o=pt208": {
                              "none/dnn": {"R": 0.136, "C": 0.326, "A": 0.996},
                              "none/kde": {"R": 0.146, "C": 0.126,
                                           "A": 0.936}},

                          "i=pt211": {
                              "none/dnn": {"R": 0.135, "C": 0.324, "A": 0.993},
                              "none/kde": {"R": 0.143, "C": 0.123,
                                           "A": 0.932}},

                          "i=pt335": {
                              "none/dnn": {"R": 0.134, "C": 0.324, "A": 0.994},
                              "none/kde": {"R": 0.144, "C": 0.124,
                                           "A": 0.934}},

                          "i=pt342": {
                              "none/dnn": {"R": 0.135, "C": 0.325, "A": 0.995},
                              "none/kde": {"R": 0.145, "C": 0.125,
                                           "A": 0.935}},

                          "i=pt208": {
                              "none/dnn": {"R": 0.136, "C": 0.326, "A": 0.996},
                              "none/kde": {"R": 0.146, "C": 0.126,
                                           "A": 0.936}}}

    # TODO: We will need to convert the queries/labels into "car, bus, van,
    #  others". This is how the dataset defines things

    label_desc = {"t": [constants.DISCRETE, ["sedan", "suv", "truck", "van"]],
                  "s": [constants.CONTINUOUS, [40, 50, 60, 65, 70]],
                  "c": [constants.DISCRETE,
                        ["white", "red", "black", "silver"]],
                  "i": [constants.DISCRETE,
                        ["pt335", "pt342", "pt211", "pt208"]],
                  "o": [constants.DISCRETE,
                        ["pt335", "pt342", "pt211", "pt208"]]}

    qo = QueryOptimizer()

    print("Running Query Optimizer Demo...")

    for query in query_list_mod:
        print(query, " -> ", (
            qo.run(query, synthetic_pp_list, synthetic_pp_stats, label_desc)))
        # print qo.run(query, synthetic_pp_list_short,
        # synthetic_pp_stats_short, label_desc)
