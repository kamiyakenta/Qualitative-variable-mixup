from enum import Enum, auto

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


class MixData(Enum):
    A_AND_A = auto()
    A_AND_B = auto()
    B_AND_B = auto()


class CategoricalMixup():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_names = X_train.columns.tolist()
        self.y_name = y_train.name
        self.all_df = pd.concat([X_train, y_train], axis=1)

    def _under_sampling_df(self, ratio_unit, u_0_size, u_1_size):
        res = RandomUnderSampler(
            ratio={0: ratio_unit*u_0_size, 1: ratio_unit*u_1_size},
            random_state=2
        )
        X_res, y_res = res.fit_sample(self.X_train, self.y_train)
        X_res, y_res = (pd.DataFrame(X_res), pd.DataFrame(y_res))
        X_res.columns, y_res.columns = (self.X_names, [self.y_name])
        return pd.concat([X_res, y_res], axis=1)

    def _random_sampling_dfs(self, df_1, df_2, quantities, categories, size):
        random_sample_dfs = []
        for df in [df_1[quantities], df_1[categories], df_2[quantities], df_2[categories]]:
            random_sample_dfs.append(df.sample(n=size).sample(frac=1))
        return random_sample_dfs

    def _mix(self, mixup_df, ratio, ratio_unit, set=MixData.A_AND_B):
        if ratio != 0:
            all_df = self.all_df
            categories = self.categories
            quantities = [col for col in all_df.columns.tolist() if not col in categories]
            if set == MixData.A_AND_A:
                a, b = (0, 0)
            elif set == MixData.A_AND_B:
                a, b = (0, 1)
            elif set == MixData.B_AND_B:
                a, b = (1, 1)

            for i in range(ratio):
                [A_q, A_c, B_q, B_c] = self._random_sampling_dfs(
                    all_df[all_df[self.y_name] == a],
                    all_df[all_df[self.y_name] == b],
                    quantities, categories, ratio_unit
                )
                lmd = np.random.beta(self.alpha, self.alpha, ratio_unit).reshape(ratio_unit, 1)
                df = pd.DataFrame(index=A_q.index, columns=all_df.columns)
                df[quantities] = lmd * A_q.values + (1-lmd) * B_q.values
                for category in categories:
                    generate_c_list = []
                    for i, (normal, fraud) in enumerate(zip(np.array(A_c[category]), np.array(B_c[category]))):
                        generate_c_list.append(np.random.choice([normal, fraud], p=[lmd[i][0], 1-lmd[i][0]]))
                    df[category] = generate_c_list
                mixup_df = pd.concat([mixup_df, df])
        return mixup_df

    def _mixup_df(self, ratio_unit, m_0_0_size, m_0_1_size, m_1_1_size):
        return self._mix(
            self._mix(
                self._mix(
                    pd.DataFrame(columns=self.all_df.columns),
                    m_0_1_size, ratio_unit, MixData.A_AND_B
                ),
                m_1_1_size, ratio_unit, MixData.B_AND_B
            ),
            m_0_0_size, ratio_unit, MixData.A_AND_A
        )

    # ratio_unit <= y_train.sum()
    def execute(self, categories, ratio_unit, ratio=[30, 0, 18, 2, 1], alpha=1.):
        self.categories = categories
        self.alpha = alpha
        data = pd.concat([
            self._under_sampling_df(ratio_unit, ratio[0], ratio[4]),
            self._mixup_df(ratio_unit, ratio[1], ratio[2], ratio[3])
        ], sort=True)
        data.reset_index(drop=True, inplace=True)
        return np.array(data[self.X_names]), np.array(data[self.y_name])
