from MDP import MDP
import pandas as pd
from Utils import powerset, value_iteration, get_policy, test_specification, rollout_policy

class Navigation(MDP):
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
        # Facts are 1. The door is closed
        # 2. The door is open
        # 3. The robot is holding the suitcase
        # 4. The robot is not holding the suitcase
        # 5. The suitcase is inside the room
        # 6. The suitcase is outside the room.
        self.fact_list = ['The door is closed', 'The door is open', 'The robot is holding the suitcase', 'The robot is not holding the suitcase', 'The suitcase is inside the room', 'The suitcase is outside the room', 'task_complete']
        self.fact_set = set(self.fact_list)

    def generate_state_space(self):
        state_space = powerset(self.fact_set)
        return list(state_space)

    def generate_actions(self):
        # 1. Open the door
        # 2. Move to the room
        # 3. Pick up the suitcase outside the room
        # 4. Dropoff the suitcase inside the room
        # 5. Exit the task.
        return ['Open the door', 'Pick up the suitcase outside the room',  'Dropoff the suitcase inside the room', 'Exit the task']

    def generate_init_state(self):
        return set(['The door is closed', 'The robot is not holding the suitcase', 'The suitcase is outside the room'])

    def get_transition_probability(self, state, action, next_state):
        if 'task_complete' in state:
            if state == next_state:
                return 1
            else:
                return 0
        if action == 'Open the door':
            if 'The door is closed' in state and 'The robot is holding the suitcase' in state:
                expected_next_state = (state - set(['The door is closed']))| set(['The door is open'])
                if expected_next_state == next_state:
                    return 1
            # If precondition are not met the task should exit
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        elif action == 'Pick up the suitcase outside the room':
            if 'The robot is not holding the suitcase' in state and 'The suitcase is outside the room' in state:
                expected_next_state = (state - set(['The robot is not holding the suitcase']))| set(['The robot is holding the suitcase'])
                if expected_next_state == next_state:
                    return 1
            # If precondition are not met the task should exit
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        # elif action == 'Move to the room':
        #     if 'The door is open' in state and 'The robot is holding the suitcase' in state and 'The suitcase is outside the room.' in state:
        #         expected_next_state = (state - set(['The suitcase is outside the room.']))| set(['The suitcase is inside the room'])
        #         if expected_next_state == next_state:
        #             return 1
        #     # If precondition are not met the task should exit
        #     else:
        #         expected_next_state = state | set(['task_complete'])
        #         if expected_next_state == next_state:
        #             return 1
        elif action == 'Dropoff the suitcase inside the room':
            if 'The door is open' in state and 'The robot is holding the suitcase' in state and 'The suitcase is outside the room' in state:
                expected_next_state = (state - set(['The robot is holding the suitcase','The suitcase is outside the room']))| set(['The robot is not holding the suitcase','The suitcase is inside the room'])
                if expected_next_state == next_state:
                    return 1
            # If precondition
            else:
                expected_next_state = state | set(['task_complete'])
                if expected_next_state == next_state:
                    return 1
        elif action == 'Exit the task':
            expected_next_state = state | set(['task_complete'])
            if next_state == expected_next_state:
                return 1
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
        # print(df)
        # print(df.shape)
        self.all_reward_matrices = []
        self.number_of_participants = df.shape[0]
        for row_id in range(1, df.shape[0]):
            # print(df.iloc[row_id, 212:240])
            # for col_id in range(212, 240):
            #     print("row_id: ", row_id, "column_id: ", col_id, "value: ", df.iloc[row_id, col_id])
            all_rewards = df.iloc[row_id,  83:107].tolist()
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
    mdp = Navigation('Latest.xlsx')
    target_trajectory = ['Pick up the suitcase outside the room', 'Open the door', 'Dropoff the suitcase inside the room', 'Exit the task']
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