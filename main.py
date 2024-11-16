import agent
from game import Game

match (input()):
    case "train":
        agent.play()
    case "play":
        game = Game()
        while True:
            game.human_step()
    case "show":
        agent.play(True, True)
    case "init":
        agent.init_cache()
