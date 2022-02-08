
import yaml


param = dict()

with open("test.yaml", "r") as stream:
    try:
        param = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

param["Simulator.numNodes"] = 21

# default_flow_style=False is needed in this case
with open("test.yaml", "w") as file:
    documents = yaml.dump(param, file, default_flow_style=False)
