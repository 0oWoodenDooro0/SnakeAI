import agent

match (input()):
    case "play":
        agent.play()
    case "init":
        agent.init_cache()
