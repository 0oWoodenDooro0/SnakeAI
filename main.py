import agent

match (input()):
    case "train":
        agent.play()
    case "play":
        agent.play(True, False)
    case "show":
        agent.play(True, True)
    case "init":
        agent.init_cache()
