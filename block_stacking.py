from MDP import MDP
import pandas as pd
from Utils import powerset, value_iteration, get_policy, test_specification, rollout_policy

class BlockStacking(MDP):
    def __init__(self, rewards_matrix_file, discount=0.99):
        self.discount = discount
        self.get_all_facts()
        self.rewards_matrix_file = rewards_matrix_file
        self.state_space = self.generate_state_space()
        self.actions = self.generate_actions()
        self.init_state = self.generate_init_state()
        self.read_rewards_excel_all_lines()
        self.V = []
        self.Q = []
        self.Policy = []

    def get_all_facts(self):
        self.fact_list = ['A on the ground', 'A on B', 'A on C', 'B on the ground', 'B on A', 'B on C', 'C on the ground', 'task_complete']
        self.fact_set = set(['A on the ground', 'B on C', 'C on the ground',
                             'A on B', 'A on C', 'B on the ground', 'B on A', 'task_complete'])


    def generate_state_space(self):
        state_space = powerset(self.fact_set)
        return list(state_space)

    def generate_actions(self):
        return ['Stack A on B', 'Swap A and B', 'Stack B on A', 'Exit the task']

    def generate_init_state(self):
        return set(['A on the ground', 'B on C', 'C on the ground'])

    def get_transition_probability(self, state, action, next_state):
        if 'task_complete' in state:
            if state == next_state:
                return 1
            else:
                return 0
        if action == 'Stack A on B':
            if 'A on the ground' in state and 'B on C' in state and 'C on the ground' in state:
                expected_next_state = (state - set(['A on the ground']))| set(['A on B'])
                if expected_next_state == next_state:
                    return 1
            # If precondition are not met the task should exit
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        elif action == 'Swap A and B':
            if 'A on the ground' in state and 'B on C' in state and 'C on the ground' in state:
                expected_next_state = (state - set(['A on the ground', 'B on C']))| set(['B on the ground', 'A on C'])
                if expected_next_state == next_state:
                    return 1
            # elif state == next_state:
            #     return 1
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        elif action == 'Stack B on A':
            if 'B on the ground' in state and 'A on C' in state and 'C on the ground' in state:
                expected_next_state = (state - set(['B on the ground']))| set(['B on A'])
                if expected_next_state == next_state:
                    return 1
            # elif state == next_state:
            #     return 1
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        elif action == 'Exit the task':
            expected_next_state = state | set(['task_complete'])
            if next_state == expected_next_state:
                    return 1
            # elif 'task_complete' in state and 'task_complete' in next_state and state == next_state:
            #     return 1
        return 0

    def get_reward(self, state, action, participant_id=0):
        if 'task_complete' in state:
            return 0

        total_reward = 0
        for fact in self.all_reward_matrices[participant_id][action]:
            if fact in state:
                total_reward += self.all_reward_matrices[participant_id][action][fact]
        return total_reward

    def read_rewards_excel_all_lines(self):
        # read by default 1st sheet of an excel file
        df = pd.read_excel(self.rewards_matrix_file)
        #print(df)
        #print(df.shape)
        self.all_reward_matrices = []
        self.number_of_participants = df.shape[0]
        for row_id in range(1,df.shape[0]):
            #print(df.iloc[row_id, 212:240])
            # for col_id in range(212, 240):
            #     print("row_id: ", row_id, "column_id: ", col_id, "value: ", df.iloc[row_id, col_id])
            # all_rewards = df.iloc[row_id, 212:240].tolist()
            all_rewards = df.iloc[row_id, 200:228].tolist()
            action_list = self.get_actions()
            fact_list = self.fact_list
            rewards_matrix = {act: {fact: 0 for fact in fact_list} for act in action_list}
            idx = 0
            for fact in fact_list:
                for act in action_list:
                    if fact != 'task_complete':
                        rewards_matrix[act][fact] = int(all_rewards[idx])
                        idx = idx + 1
            self.all_reward_matrices.append(rewards_matrix)


if __name__ == '__main__':
    # mdp = BlockStacking('4.0 Prolific - Goals vs Rewards - Specify Objective_February 7, 2025_13.32.xlsx')
    mdp = BlockStacking('5.0 Prolific - Goals vs Rewards - Specify Objective_February 9, 2025_19.10.xlsx')
    target_trajectory = ['Swap A and B', 'Stack B on A', 'Exit the task']
    for participant_id in range(len(mdp.all_reward_matrices)):
        value_iteration(mdp, participant_id=participant_id)
        print("Participant ID: ", participant_id)
        correct_flag, underspecified_flag = test_specification(mdp, target_trajectory, participant_id=participant_id)
        policy = get_policy(mdp,participant_id=participant_id)
        policy_rollout = rollout_policy(mdp, policy, participant_id=participant_id)
        print("Policy Rollout: ", policy_rollout)
        if correct_flag:
            print("Specification is Correct")
        if underspecified_flag:
            print("Specification is Underspecified")
