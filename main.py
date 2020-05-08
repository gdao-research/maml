from maml.learner import Learner


if __name__ == '__main__':
    vanilla = Learner()
    vanilla.learn_vanilla()
    vanilla.eval_vanilla()

    maml = Learner()
    maml.learn_maml()
    maml.eval_maml()
