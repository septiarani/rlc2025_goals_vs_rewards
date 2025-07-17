from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    pow_set = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    power_set_with_set = [set(x) for x in pow_set]
    return power_set_with_set

def test_specification(mdp, trajectory, participant_id=0):
    initial_state = mdp.get_init_state()
    current_state = initial_state
    correct_flag = True
    underspecified_flag = False
    for act in trajectory:
        max_value = -1000
        max_value_act_list = []
        for a in mdp.get_actions():
            curr_val = mdp.Q[participant_id][mdp.get_state_hash(current_state)][a]
            if curr_val > max_value:
                max_value = curr_val
                max_value_act_list = [a]
            elif curr_val == max_value:
                max_value_act_list.append(a)
        if mdp.Q[participant_id][mdp.get_state_hash(current_state)][act] != max_value:
            return False, False
        if len(max_value_act_list) > 1:
            underspecified_flag = True
            print ("Underspecified: ", current_state, act, max_value_act_list)
        for s_prime in mdp.get_state_space():
            if mdp.get_transition_probability(current_state, act, s_prime) > 0:
                current_state = s_prime
                break
    return correct_flag, underspecified_flag

def rollout_policy(mdp, policy, max_steps=1000, participant_id=0, end_token=None):
    # pass
    trajectory = []
    s = mdp.get_init_state()
    for _ in range(max_steps):
        a = policy[mdp.get_state_hash(s)]
        trajectory.append(a)
        if a == "None":
            break
        for s_prime in mdp.get_state_space():
            if mdp.get_transition_probability(s, a, s_prime) > 0:
                #print(s, a, mdp.get_reward(s, a, participant_id, print_flag=True))
                s = s_prime
                break
        if a == "Exit the task":
            break
        if end_token is not None and end_token in s_prime:
            break
        #s = mdp.get_next_state(s, a)
    return trajectory

def value_iteration(mdp, epsilon=0.001, participant_id=0):
    mdp.V.append({mdp.get_state_hash(s): 0 for s in mdp.get_state_space()})
    mdp.Q.append({mdp.get_state_hash(s): {} for s in mdp.get_state_space()})
    while True:
        delta = 0
        for s in mdp.get_state_space():
            s_hash = mdp.get_state_hash(s)
            v = mdp.V[participant_id][s_hash]
            # for R(s, a)
            curr_max =-1000
            for a in mdp.get_actions():
            #     for s_old in mdp.get_state_space():
            #         count = 0
            #         max_reward = -1000
            #         for s_prime in mdp.get_state_space():
            #             if mdp.get_transition_probability(s_old, a, s_prime) > 0:
            #                 count += 1
            #                 max_reward = max(max_reward, mdp.get_reward(s_old, a))
            #                 #print("s: ", s_old, "a: ", a, "s_prime: ", s_prime, "R: ", mdp.get_reward(s_old, a), "P: ", mdp.get_transition_probability(s_old, a, s_prime))
            #         print("s: ", s_old, "a: ", a, "count: ", count)#, "max_reward: ", max_reward)
            #
            # exit(0)
                mdp.Q[participant_id][s_hash][a] = mdp.get_reward(s, a, participant_id) + sum([ mdp.get_transition_probability(s, a, s_prime) *
                                        (mdp.discount * mdp.V[participant_id][mdp.get_state_hash(s_prime)])
                                        for s_prime in mdp.get_state_space()])
                curr_max = max(curr_max, mdp.Q[participant_id][s_hash][a])
                #print("s: ", s, "a: ", a, "Q: ", mdp.Q[s_hash][a])
            mdp.V[participant_id][s_hash] = curr_max
            # for R(s, a, s')
            # V[s_hash] = max([mdp.discount * sum([mdp.get_transition_probability(s, a, s_prime) * (mdp.get_reward(s, a, s_prime) + V[mdp.get_state_hash(s_prime)]) for s_prime in mdp.get_state_space()]) for a in mdp.get_actions()])
            delta = max(delta, abs(v - mdp.V[participant_id][s_hash]))
        # print(delta)
        if delta < epsilon:
            # self.delta = delta
            break
    return mdp.V, mdp.Q

def value_iteration_sas(mdp, epsilon=0.001, participant_id=0):
    mdp.V.append({mdp.get_state_hash(s): 0 for s in mdp.get_state_space()})
    mdp.Q.append({mdp.get_state_hash(s): {} for s in mdp.get_state_space()})
    while True:
        delta = 0
        for s in mdp.get_state_space():
            s_hash = mdp.get_state_hash(s)
            v = mdp.V[participant_id][s_hash]
            # for R(s, a)
            curr_max =-1000
            for a in mdp.get_actions():
            #     for s_old in mdp.get_state_space():
            #         count = 0
            #         max_reward = -1000
            #         for s_prime in mdp.get_state_space():
            #             if mdp.get_transition_probability(s_old, a, s_prime) > 0:
            #                 count += 1
            #                 max_reward = max(max_reward, mdp.get_reward(s_old, a))
            #                 #print("s: ", s_old, "a: ", a, "s_prime: ", s_prime, "R: ", mdp.get_reward(s_old, a), "P: ", mdp.get_transition_probability(s_old, a, s_prime))
            #         print("s: ", s_old, "a: ", a, "count: ", count)#, "max_reward: ", max_reward)
            #
            # exit(0)
                mdp.Q[participant_id][s_hash][a] = sum([ mdp.get_transition_probability(s, a, s_prime) *
                                        (mdp.discount * (mdp.get_reward(s, a,s_prime , participant_id)+ mdp.V[participant_id][mdp.get_state_hash(s_prime)]))
                                        for s_prime in mdp.get_state_space()])
                curr_max = max(curr_max, mdp.Q[participant_id][s_hash][a])
                #print("s: ", s, "a: ", a, "Q: ", mdp.Q[s_hash][a])
            mdp.V[participant_id][s_hash] = curr_max
            # for R(s, a, s')
            # V[s_hash] = max([mdp.discount * sum([mdp.get_transition_probability(s, a, s_prime) * (mdp.get_reward(s, a, s_prime) + V[mdp.get_state_hash(s_prime)]) for s_prime in mdp.get_state_space()]) for a in mdp.get_actions()])
            delta = max(delta, abs(v - mdp.V[participant_id][s_hash]))
        print(delta)
        if delta < epsilon:
            # self.delta = delta
            break
    return mdp.V, mdp.Q

def get_policy(mdp, participant_id=0):
    # pass
    # initialize P with None
    mdp.Policy.append({mdp.get_state_hash(s): None for s in mdp.get_state_space()})
    for s in mdp.get_state_space():
        s_hash = mdp.get_state_hash(s)
        max_value = float('-inf')
        best_action = None
        for a in mdp.get_actions():
            value = mdp.Q[participant_id][s_hash][a]
            if value >= max_value:
                max_value = value
                best_action = a
            # How if there are multiple actions that give max expected value?
            # Currently, select the action in the latest order
            # Next, should be able to get all possible trajectories
        mdp.Policy[participant_id][s_hash] = best_action
    mdp.Policy[participant_id]['Terminate'] = "None"
    # self.P = P
    # print("delta: ", delta)
    # print("V: ", V)
    # print("P: ", P)
    return mdp.Policy[participant_id]