import os
import pickle
import json


class file_generating:

    def load_from_file(self, filename):
        if type(filename) == str:
            model_file = filename
        else:
            model_file = filename.name
        if os.stat(model_file).st_size == 0:
            print("Error loading CF SVD model")
            exit(1)
        try:
            file = open(model_file, "rb")
        except IOError:
            print("There is no model file")
        u_bias = pickle.load(file)
        W_user = pickle.load(file)
        i_bias = pickle.load(file)
        W_item = pickle.load(file)
        g_bias = pickle.load(file)
        tmp_ufactor = W_user[0]
        tmp_ifactor = W_item[0]
        u_val = []
        i_val = []
        u_id = []
        i_id = []
        u_value = []
        i_value = []
        u_bias_values = []
        i_bias_values = []
        with open("ua.base.feature", "r") as file:
            for line in file:
                line_value = line.split()
                u_val.append(line_value[4])
                i_val.append(line_value[5])
        for i in range(len(u_val)):
            u, v = u_val[i].split(":")
            u_id.append(int(u))
            u_value.append(int(v))
            u_bias_values.append(u_bias[int(u)])
        for j in range(len(i_val)):
            x, y = i_val[j].split(":")
            i_id.append(int(x))
            i_value.append(int(y))
            i_bias_values.append(i_bias[int(x)])
        user_factor = W_user.tolist()
        item_factor = W_item.tolist()
        write_output_test = open("test_input_mrjob", "w")
        json.dump([list(u_value), list(i_value)], write_output_test)
        write_output_model = open("model_input_mrjob", "w")
        json.dump([list(u_bias_values), list(i_bias_values), user_factor, item_factor], write_output_model)

if __name__ == "__main__":
    fi = "0100.model"
    file_generating().load_from_file(fi)
