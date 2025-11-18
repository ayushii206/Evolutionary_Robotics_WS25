### (b) Why this problem is “good-natured”

This optimization task has a *smooth fitness landscape* because each character in the target string contributes independently to the total fitness.  
Changing one character can only improve or leave unchanged the fitness of other positions—there are no deceptive interactions.  
Thus, the number of correct letters tends to increase steadily, producing monotonic progress without local minima.  

Small random mutations have a high probability of generating slight improvements, and every intermediate step gives feedback toward the goal.  
This makes the landscape **unimodal** and **gradient-like**, so a simple hill-climber reliably converges to the target within a few thousand generations.
