import re
import random
import numpy as np
import os
import math

class Dataset(object):


    def __init__(self, data=list(), target=list(), target_names=list(), feature_names=list()):
        self.data          = list(data)
        self.target        = list(target)
        self.target_names  = list(target_names)
        self.feature_names = list(feature_names)
        self.missing_data  = {}

    def add_to_data(self, data, target, target_names):
        for datapoint in data:
            self.data.append(datapoint)

        for targetpoint in target:
            self.target.append(targetpoint)

        for new_target_name in target_names:
            dont_add = False
            for target_name in self.target_names:
                if target_name == new_target_name:
                    dont_add = True
            if not dont_add:
                self.target_names.append(new_target_name)

    def set_missing_data(self):
        for datapoint_index, column_index in self.missing_data.items():
            self.data[datapoint_index][column_index] = -1

    def discretize(self):
        # CREDIT GOES TO PRESTON
        bins = np.linspace(0, 1, num=10)

        for i in range(len(self.feature_names)):
            self.data[:, i] = np.digitize([index[i] for index in self.data], bins)

    def randomize(self):

        data_list = []
        target_list = []
        index_list = list(range(0, self.data.__len__()))
        random.shuffle(index_list)

        for index in range(index_list.__len__()):
            data_list.append(self.data[index_list[index]])
            target_list.append(self.target[index_list[index]])

        self.data = data_list
        self.target = target_list

    def split_dataset(self, split_percentage):

        length = self.data.__len__()
        top_training_index = math.floor(length * split_percentage)
        set_1_data = []
        set_1_target = []
        set_2_data = []
        set_2_target  = []

        for index in range(0, top_training_index):
            set_1_data.append([item for item in self.data[index]])
            set_1_target.append(self.target[index])

        for index in range(top_training_index, length):
            set_2_data.append([item for item in self.data[index]])
            set_2_target.append(self.target[index])

        set_1_dataset = Dataset(set_1_data, set_1_target, self.target_names, self.feature_names)
        set_2_dataset = Dataset(set_2_data, set_2_target, self.target_names, self.feature_names)

        return [ set_1_dataset, set_2_dataset ]

    def read_file_into_dataset(self, data_set_name):
        target_names_dict, target_info = self.__read_names_file(data_set_name)
        self.__read_data_file(data_set_name, target_names_dict, target_info)

    def __read_data_file(self, data_set_name, target_names_dict, target_info):
        data_file = data_set_name + ".data"

        with open(data_file) as data_info:
            for line in data_info:
                pattern = re.compile(r'\s+')
                line = re.sub(pattern, '', line)

                single_data = line.split(',')

                for index in range(single_data.__len__() - 1):
                    if "continuous" not in target_info[index] and single_data[index] != "?":
                        single_data[index] = target_info[index][single_data[index]]
                    elif single_data[index] == "?":
                        self.missing_data[self.data.__len__()] = index
                        single_data[index] = 0

                self.target.append(target_names_dict[single_data.pop()])
                for i in range(single_data.__len__()):
                    single_data[i] = float(single_data[i])
                self.data.append(single_data)

    def __read_names_file(self, data_set_name):
        name_file = data_set_name + ".names"

        target_info = []
        target_names_dict = {}

        numeric_dic = self.__read_numeric_file(data_set_name)

        line_number = 1
        with open(name_file) as name_info:
            for line in name_info:
                pattern = re.compile(r'\s+')
                line = re.sub(pattern, '', line)

                # Valid line?
                if line is not '' and line[0] is not '|':
                    # Classes line or attributes line
                    if line_number is 1:
                        # Get rid of all spaces
                        line = re.sub(pattern, '', line)

                        # Get the target/classes names
                        self.target_names = line.split(',')

                        # Transfer target/classes names to a dictionary
                        for index in range(self.target_names.__len__()):
                            target_names_dict[self.target_names[index]] = index

                        line_number += 1
                    else:
                        # Get the attribute names and possible values.
                        attribute_info = line.split(':')
                        attribute_values = attribute_info[1].replace('.', '').split(',')
                        self.feature_names.append(attribute_info[0])

                        # Handle Nominal Data
                        target_info.append(self.__handle_nominal_data(attribute_values, data_set_name, numeric_dic))
        return target_names_dict, target_info

    def __read_numeric_file(self, data_set_name):
        numeric_dic = {}

        numeric_file = data_set_name + ".numeric"
        if os.path.isfile(numeric_file):
            with open(numeric_file) as numeric_info:
                for line in numeric_info:
                    pattern = re.compile(r'\s+')
                    line = re.sub(pattern, '', line)

                    if line is not '':
                        attribute_info = line.split(':')
                        numeric_dic[attribute_info[0]] = attribute_info[1].replace('.', '').split(',')

        return numeric_dic

    def __handle_nominal_data(self, attribute_values, data_set_name, numeric_dict: dict):
        attribute_dict = {}
        integer = 0

        for index in range(attribute_values.__len__()):
            if numeric_dict.__len__() != 0:
                attribute_dict[attribute_values[index]] = numeric_dict[self.feature_names[self.feature_names.__len__()-1]][index]
            elif attribute_values[index] == "continuous":
                return { "continuous": "continuous" }
            elif attribute_values[index].isnumeric():
                attribute_dict[attribute_values[index]] = int(attribute_values[index])
                integer = int(attribute_values[index]) + 1
            else:
                attribute_dict[attribute_values[index]] = integer
                integer += 1

        return attribute_dict
