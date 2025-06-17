# Logging utility

def log(message):
    with open('logs/agent.log', 'a') as f:
        f.write(message + '\n')
    print(message)