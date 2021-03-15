import azureml.core
from azureml.core import workspace

if __name__ == '__main__':
    print("I'm working")
    ws = workspace.from_config()
    print(ws.name, ws.resource_group, ws.location, ws.subscription, sep='\n')