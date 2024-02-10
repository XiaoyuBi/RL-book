import matplotlib.pyplot as plt

class Game:
    def __init__(self, p: float = 0.5, gamma: float = 1.0):
        self.p = p # probabiity of winning
        self.gamma = gamma # discounting factor

        # states {1, 2, 3, ..., 99}
        # two terminal states: state 0 for reward 0 and state 100 for reward 1
        self.states = [i for i in range(0, 101)]
        # actions {0, 2, ..., min(s, 100 - s)}
        self.actions = {}
        for s in self.states:
            self.actions[s] = list(range(1, min(s, 100 - s) + 1))
        
        # Value Function init
        self.val_func = [0] * len(self.states)
        self.logs = [self.val_func]
        # Policy init
        self.policy = [1] * len(self.states)
    
    def value_iteration(self):
        # V(0) and V(100) will always be 0, since they are terminal states
        new_val_func = [0] * len(self.states)
        for s in range(1, 100):

            # find max val_func result with iterating througn actions
            for a in self.actions[s]:
                # toss the coin
                r = 1 if s + a >= 100 else 0
                res = self.p * (r + self.gamma * self.val_func[s + a]) + \
                    (1 - self.p) * (0 + self.gamma * self.val_func[s - a])
                
                new_val_func[s] = max(new_val_func[s], res)
        
        # after value iteration
        self.val_func = new_val_func
        self.logs.append(self.val_func)
    
    def policy_iteration(self):
        # policy evaluation
        # V(0) and V(100) will always be 0, since they are terminal states
        new_val_func = [0] * len(self.states)
        for s in range(1, 100):
            a = self.policy[s]
            r = 1 if s + a >= 100 else 0
            res = self.p * (r + self.gamma * self.val_func[s + a]) + \
                (1 - self.p) * (0 + self.gamma * self.val_func[s - a])

            new_val_func[s] = res
        
        self.val_func = new_val_func
        self.logs.append(self.val_func)

        # policy improvement
        for s in range(1, 100):
            # for each state, find the greedy action
            best_val = self.val_func[s]
            best_action = self.policy[s]
            for a in self.actions[s]:
                r = 1 if s + a >= 100 else 0
                res = self.p * (r + self.gamma * self.val_func[s + a]) + \
                    (1 - self.p) * (0 + self.gamma * self.val_func[s - a])
                
                if res > best_val:
                    best_val = res
                    best_action = a
            
            self.policy[s] = best_action


if __name__ == "__main__":
    # p = 0.25 with value iteration
    game = Game(p = 0.25)
    for i in range(30):
        game.value_iteration()
    
    plt.title("p = 0.25 with value iteration")
    plt.plot(game.logs[0][:-1], label = "iteration 0") 
    plt.plot(game.logs[5][:-1], label = "iteration 5") 
    plt.plot(game.logs[10][:-1], label = "iteration 10") 
    plt.plot(game.logs[30][:-1], label = "iteration 30") 
    plt.legend()
    plt.savefig("./programming/Ex4.9/fig1.jpg")
    plt.close()

    # p = 0.55 with policy iteration
    game = Game(p = 0.55)
    for i in range(30):
        game.policy_iteration()
    
    plt.title("p = 0.55 with policy iteration")
    plt.plot(game.logs[0][:-1], label = "iteration 0") 
    plt.plot(game.logs[5][:-1], label = "iteration 5") 
    plt.plot(game.logs[10][:-1], label = "iteration 10") 
    plt.plot(game.logs[30][:-1], label = "iteration 30") 
    plt.legend()
    plt.savefig("./programming/Ex4.9/fig2.jpg")
    plt.close()
