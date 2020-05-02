from mrjob.compat import jobconf_from_env
from mrjob.job import MRJob
from mrjob.step import MRStep
import json

base_score = 3.0


class infer(MRJob):

    def configure_args(self):
        super(infer, self).configure_args()
        self.add_passthru_arg("--test-input", default="test", dest="test")

    def decide_input_file(self):
        # return 1 if test.feature , return 2 if mapper is reading .model file.
        filename = jobconf_from_env(
            "map.input.file")  # this is to get the name of the file a mapper is reading input from.
        if self.options.test in filename:
            return 1
        else:
            return 2

    def mapping_values(self, _, line):
        file = self.decide_input_file()
        if file == 1:
            line_val = json.loads(line)
            for i in range(len(line_val[0])):
                u_val = int(line_val[0][i])
                u_id = int(i)
                yield u_id, (0, u_id, u_val)

            for j in range(len(line_val[1])):
                i_val = int(line_val[1][j])
                i_id = int(j)
                yield i_id, (2, i_id, i_val)

        else:
            li = json.loads(line)
            u_bias = li[0]
            i_bias = li[1]
            user_factor = li[2][0]
            item_factor = li[3][0]
            for i in range(len(u_bias)):
                u_bias_val = u_bias[i]  # gets the value from u_bias
                u_id = int(i)
                yield u_id, (1, u_id, u_bias_val)
            for j in range(len(i_bias)):
                i_bias_val = i_bias[j]
                i_id = int(j)
                yield i_id, (3, i_id, i_bias_val)
            for k in range(len(user_factor)):
                user_factor_val = user_factor
                user_factor_id = int(k)
                yield user_factor_id, (4, user_factor_id, user_factor_val)
            for m in range(len(item_factor)):
                item_factor_val = item_factor
                item_factor_id = int(m)
                yield item_factor_id, (5, item_factor_id, item_factor_val)

    def reducing_values(self, id, values):
        values_of_user_test_file = []
        values_of_user_model_file = []
        values_of_item_test_file = []
        values_of_item_model_file = []
        values_of_user_factor = []
        values_of_item_factor = []
        for val in values:
            if val[0] == 0:
                values_of_user_test_file.append(val)
            if val[0] == 1:
                values_of_user_model_file.append(val)
            if val[0] == 2:
                values_of_item_test_file.append(val)
            if val[0] == 3:
                values_of_item_model_file.append(val)
            if val[0] == 4:
                values_of_user_factor.append(val)
            if val[0] == 5:
                values_of_item_factor.append(val)

        for i in range(len(values_of_user_test_file)):
            user = values_of_user_model_file[0][2]
            user_key = values_of_user_model_file[0][1]
            user_score = values_of_user_test_file[i][2]
            final_score_user = user_score * user
            yield user_key, final_score_user
        for j in range(len(values_of_item_test_file)):
            item = values_of_item_model_file[0][2]
            item_key = values_of_item_model_file[0][1]
            item_score = values_of_item_test_file[j][2]
            final_score_item = item_score * item
            yield item_key, final_score_item
        for n in range(len(values_of_user_factor)):
            for m in range(len(values_of_item_factor)):
                user_feature = values_of_user_test_file[n][2]
                user_latent_factor = values_of_user_factor[0][2]
                #user_latent_factor_new = np.multiply(user_latent_factor, user_feature)
                user_latent_factor_new = [i * user_feature for i in user_latent_factor]
                user_factor_key = values_of_user_factor[0][1]
                item_feature = values_of_item_test_file[m][2]
                item_latent_factor = values_of_item_factor[0][2]
                #item_latent_factor_new = np.multiply(item_latent_factor, item_feature)
                item_latent_factor_new = [i * item_feature for i in item_latent_factor]
                final = sum(i*j for i,j in zip(user_latent_factor_new,item_latent_factor_new))
                #final = np.dot(user_latent_factor_new, item_latent_factor_new)
                yield user_factor_key, final

    def mapping_values_keys(self, k, v):
        yield k, v

    def sum_of_values_by_keys(self, k, values):
        yield k, sum(values) + base_score

    def steps(self):
        return [
            MRStep(mapper=self.mapping_values, reducer=self.reducing_values),
            MRStep(mapper=self.mapping_values_keys, reducer=self.sum_of_values_by_keys)
        ]


if __name__ == "__main__":
    infer.run()
